package cuttingedge.ml

import com.doofin.stdScala.lg
import org.bytedeco.pytorch.{LinearImpl, Tensor, TensorArrayRef, TensorVector}
//import org.bytedeco.javacpp._
//import org.bytedeco.pytorch._
//import org.bytedeco._
import cuttingedge.ml.torchUtils._
import org.bytedeco.pytorch
import org.bytedeco.pytorch.Module
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.{torch => th}
import org.bytedeco.pytorch.global.torch._
import org.bytedeco.pytorch.EmbeddingImpl
import org.bytedeco.pytorch.LSTMImpl
import org.bytedeco.pytorch.LSTMOptions
import org.bytedeco.pytorch.GRUImpl
import org.bytedeco.pytorch.GRUOptions
import scala.collection.mutable.ArrayBuffer

/** ku ATNLP 
  Assignment3 grp proj Dec15
  Assignment4 indivi proj Jan20

Generalization without Systematicity: On the Compositional Skills
of Sequence-to-Sequence Recurrent Networks by Brendan Lake and Marco Baroni

Luong or Badahnau use DECODER hidden state and the ENCODER output to calculate the weight, 
however Pytorch tutorial use only DECODER input and DECODER hidden state to calculate the weight
  */

object seq2seqLakeAttn {

  //
  case class decoderAttnBahd(
      hidden_size: Long = 100,
      output_size: Long,
      // mName: String = "_decode",
      n_layers: Int = 1
  ) extends Module {
    val linearAttn = register_module("linearAttn", new LinearImpl(hidden_size, hidden_size))
    val linearAttnCombine = register_module("linearComb", new LinearImpl(hidden_size * 2, hidden_size))
    val linearOut = register_module("linearOut", new LinearImpl(hidden_size, output_size))

    val emb = register_module("emb", new EmbeddingImpl(output_size, hidden_size))
    val v: Tensor = register_parameter("v", torch.rand(hidden_size)) //(hidden_size, 1)
    val W: Tensor = register_parameter("W", torch.rand(hidden_size, hidden_size))
    val U: Tensor = register_parameter("U", torch.rand(hidden_size, hidden_size))

    /* val rnn: LSTMImpl = register_module(
      "lstm",
      new LSTMImpl(new LSTMOptions(hidden_size, hidden_size) { num_layers().put(n_layers) })
    ) */

    //  The hidden state for the LSTM is a tuple containing both the cell state and the hidden state,
    // whereas the GRU only has a single hidden state.
    val rnnGru = register_module("gru", new GRUImpl(new GRUOptions(hidden_size, hidden_size)))

    def e_ij(prevDecHidden: Tensor, j_encHidden_k: Tensor) = { //query,key
      // (1,1,hid) + (1,1,hid)

      // v.t() matrixMul torch.tanh((W.mm(g) + torch.mm(W, g))) // yes you can always use @ in place of .matmul
      val lineared_sum =
        (W matrixMul prevDecHidden.squeeze()) + (U matrixMul j_encHidden_k.squeeze()) //(hid)
      // v.dot(linear_attn.forward(concatR)) //(hidSz,1) dot (1,1,hid) (hidSz,hidSz) = scalar?
      v dot (th.tanh(lineared_sum)) // scalar
    }

    def a_ij_weights(prevDecHidden: Tensor, encoderHiddens: Seq[Tensor]) = { //j_encHidden: Tensor,
      val stacked_squ =
        th.stack(encoderHiddens)
          .squeeze() //(1,1,hid) => (len,1,1,hid) , sfmx with respect to all enc hiddens
      // lg("stacked squ shape : ", stacked_squ.shape().toList)
      val sfmx =
        stacked_squ.softmax(0) //encoderHiddens.map(j => th.softmax(e_ij(prevDecHidden, j), 1))
      tensor2tensorSeq(sfmx) //(encHidLen,hid) to list of vec[hid] , len = encHidLen
    }
    def contextF(prevDecHidden: Tensor, encoderHiddens: Seq[Tensor]) = { //shape = encoderHidden
      //encoderHiddens is value in q k v
      val r = a_ij_weights(prevDecHidden, encoderHiddens) // encoder_hiddens :seq of (1,1,hidsz)
        .zip(encoderHiddens)
        .map { case (a_ij_scalar, h_t) =>
          // printTensor(scalar, "scalar", true)
          // printTensor(h_t, " h_t", true)
          // scalar * h_t
          h_t mulScalar a_ij_scalar
        }
        .reduce(_ + _)
      r
    }
    def forwardOnce(input: Tensor, prevHidden: Tensor, encoder_hiddens: Seq[Tensor]) = {
      val x_embed = th.dropout(emb.forward(input).view(1, 1, -1)) //1,1,hiddenSize

      val ctx = contextF(prevHidden, encoder_hiddens) // 1,1,hidsz
      // printTensor(ctx, "ctx") // 1,1,hidsz
      // hidden = torch.concat((input_hidden, c_i), dim=2)
      // # Concatenate the context vector and the decoder hidden state
      val combined =
        th.stack(Array(x_embed, ctx))
          .view(-1) // stack((1,1,hidsz,1,1,hidsz))=2,1,1,hidsz,by view -1 then result is 2*hidsz
      val combinedAttn = linearAttnCombine.forward(combined).unsqueeze(0).unsqueeze(0) // hidsz to 1,1,hidsz
      val (out0, hid) = tensorTuple2tup(rnnGru.forward(combinedAttn.relu(), prevHidden))
      val out = out0.get(0) //(1, hid)

      // printTensor(out, "out", printContent = true)
      val outProj = linearOut.forward(out)
      // printTensor(outProj, "outProj", true)
      val softmaxed = outProj.log_softmax(1)
      // printTensor(softmaxed, "softmaxed", true)
      // assert(false)
      (softmaxed, hid)
    }
    def getSeqLoss(
        inputInit: Tensor,
        encHiddens: Seq[Tensor],
        y_targetSeq: Seq[Tensor],
        teacher_forcing: Boolean = false
    ) = {
      var hidVar: Tensor = encHiddens.reverse.head
      var lossAccumVar = th.zeros(
        Array(1L),
        new pytorch.TensorOptions(torch.ScalarType.Float).requires_grad(new pytorch.BoolOptional(true))
      )
      var inputVar = inputInit

      y_targetSeq.foreach { y_dataset_target =>
        val (modelOut, hiddenOut) = forwardOnce(inputVar, hidVar, encHiddens)
        hidVar = hiddenOut
        inputVar = if (teacher_forcing) {
          y_dataset_target // # Teacher forcing: Feed the target as the next input
        } else {
          val (topK, topI) = tensorTuple2tup(
            modelOut.topk(1)
          ) // # Without teacher forcing: use its own predictions as the next input
          topI.squeeze().detach()
        }
        val y_modelOut = modelOut // .squeeze(0L) //alreay squeezed in fwOnce (get0)
        // printTensor(y_modelOut, "y_modelOut", true)
        // printTensor(y_dataset_target, "y_dataset_target", true)
        val loss =
          th.nll_loss(y_modelOut, y_dataset_target) // y_dataset_target o.squeeze() rm 1s in nth dim
        // println("loss", loss.item_float())
        /*  :
  (------tensor y_modelOut shape:,List(1, 8),-------)
 0.1313  0.1166  0.1406  0.1249  0.1289  0.1172  0.1204  0.1202
 : (------tensor y_dataset_target shape:,List(1),-------)
[ CPUFloatType{1,8} ] 1
(loss,-0.1166186) */
        // assert(false)
        // loss.print()
//        losses ++= Array(loss)
        lossAccumVar = loss + lossAccumVar

      }
      lossAccumVar
    }

    def forwardEval(
        init: Tensor,
        encHiddens: Seq[Tensor],
        max_length: Int,
        eos: Int //EOS
    ): Array[Long] = {
      //      val init = (t.head, initHid())
      //      t.tail.foldLeft(init) { (hid, input) => md.forward(input, hid) }
      var hidVar: Tensor = encHiddens.reverse.head
      val outs = new ArrayBuffer[Long]
      var inputVar = init
      import scala.util.control.Breaks._

      breakable {

        (0 to max_length).foreach { _ =>
          // printTensor(inputVar, "inputVar")
          // printTensor(hidVar.get0(), "hidVar")

          val (out, hid) = forwardOnce(inputVar, hidVar, encHiddens)
          hidVar = hid

          val idxTensor = out.topk(1).get1.squeeze().detach()

          inputVar = idxTensor
          val idx = idxTensor.item_long()
          outs.append(idx)
          if (idx == eos) break
        }
      }

      outs.toArray
    }
    // different from seq2seq tutorial !  encoder hidden states, h1 , . . . , hT , not encoder output
  }

  case class encoderGRU(
      input_size: Long,
      hidden_size: Long = 100,
      mName: String = "_encode",
      n_layers: Int = 1
  ) extends Module {
    val emb = register_module("emb" + mName, new pytorch.EmbeddingImpl(input_size, hidden_size))
    val rnnGru =
      register_module("gru", new pytorch.GRUImpl(new pytorch.GRUOptions(hidden_size, hidden_size)))

    def getInitHid() = {
      // h_n: tensor of shape (D∗num_layers,Hout)
      torch.zeros(n_layers, 1, hidden_size)
    }

    def forwardOnce(input: Tensor, hidden: Tensor) = {
      // x->embed->dropout->lstm
      val x_embed = emb.forward(input).view(1, 1, -1) //1,1,hiddenSize
      rnnGru.forward(x_embed, hidden)
    }

    /**
      * @param tensorList
      * @return list Hidden
      */
    def forwardList(tensorList: Seq[Tensor]): Seq[Tensor] = {
      //      t.tail.foldLeft(init) { (hid, input) => md.forward(input, hid) }
      var hidVar = getInitHid()
      val hiddens = new ArrayBuffer[Tensor]
      tensorList.foreach { x =>
        val (o, hidOut) = tensorTuple2tup(forwardOnce(x, hidVar))
        hiddens.append(hidOut)
        hidVar = hidOut
      }
      hiddens.toSeq
    }
  }

}

/*
class DecoderCell(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(DecoderCell, self).__init__()
        self.rnn = nn.RNN(1, embedding_dim)
        self.W = torch.nn.Parameter(torch.randn((embedding_dim, embedding_dim)))
        self.U = torch.nn.Parameter(torch.randn((embedding_dim, embedding_dim)))
        self.v = torch.nn.Parameter(torch.randn((embedding_dim, 1)))
        self.nonlinear = torch.nn.Tanh()

    def e(self, g, h):
        h = torch.tensor([h]).unsqueeze(1)
        return self.v.T @ self.nonlinear(self.W * g + self.U * h)

    def alpha(self, encoder_hiddens, input_hidden, t):
        T = len(encoder_hiddens)
        top = torch.exp(self.e(input_hidden, encoder_hiddens[t]))

        bottom = 0

        for j in range(T):
            bottom += torch.exp(self.e(input_hidden, encoder_hiddens[j]))

        return top/bottom

    def forward(self, x, encoder_hiddens, input_hidden):
        c_i = 0

        for t in range(len(encoder_hiddens)):
            alpha_it = self.alpha(encoder_hiddens, input_hidden, t)
            h_t = encoder_hiddens[t]
            c_i += alpha_it * h_t

        _, hidden = self.rnn(x, c_i)
        hidden = torch.concat((hidden, c_i), dim=1).squeeze()
        prediction = torch.argmax(F.softmax(hidden, dim=0))

        return prediction, hidden

 */
