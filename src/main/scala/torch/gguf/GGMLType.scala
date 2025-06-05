package torch.gguf

import torch.gguf.GGMLType.{getBlockByteSize, getElementsPerBlock}

import java.util.stream.LongStream
enum GGMLType:
  case F32, F16, Q4_0, Q4_1, Q4_2,   Q4_3, Q5_0, Q5_1, Q8_0, Q8_1, 
  Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, I8, I16, I32, I64, F64, IQ1_M, BF16, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8, TQ1_0, TQ2_0

  def byteSizeFor(numberOfElements: Long): Long = {
    val t = numberOfElements * getBlockByteSize
    assert(t % getElementsPerBlock == 0)
    t / getElementsPerBlock
  }

  private def numberOfElements(dims: Long*) = {
    assert(LongStream.of(dims).allMatch((d: Long) => d > 0))
    LongStream.of(dims).reduce(1, Math.multiplyExact)
  }
  def byteSizeForShape(dims: Long*): Long = byteSizeFor(numberOfElements(dims*))

object GGMLType {
//  type GGMLType = Value
//  val F32, F16, Q4_0, Q4_1, Q4_2, // support has been removed
//  Q4_3, // support has been removed
//  Q5_0, Q5_1, Q8_0, Q8_1, // k-quantizations
//  Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, I8, I16, I32, I64, F64, IQ1_M, BF16, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8, TQ1_0, TQ2_0 = Value
//  
  private val FLOAT16_BYTES = 2
  private val BFLOAT16_BYTES = 2
//  private val VALUES = valuesprivate
  val blockByteSize = 0
  private val elementsPerBlock = 0
  private val isQuantized = false
  def getBlockByteSize: Int={
    return blockByteSize
  }

  def getElementsPerBlock: Int = elementsPerBlock

//  def isQuantized(): Boolean = isQuantized

  def fromId(id: Int): GGMLType = this match {
    case F32 => F32
    case F16 => F16
    case Q4_0 => Q4_0
    case Q4_1 => Q4_1
    case Q4_2 => Q4_2
    case Q4_3 => Q4_3
    case Q5_0 => Q5_0 
    case Q5_1 => Q5_1
    case Q8_0 => Q8_0
    case Q8_1 => Q8_1
    case Q2_K => Q2_K
    case Q3_K => Q3_K
    case Q4_K => Q4_K
    case Q5_K => Q5_K
    case Q6_K => Q6_K
    case Q8_K => Q8_K
    case I8 => I8
    case I16 => I16
    case I32 => I32
    case I64 => I64
    case F64 => F64
    case IQ1_M => IQ1_M
    case BF16 => BF16
    case Q4_0_4_4 => Q4_0_4_4
    case Q4_0_4_8 => Q4_0_4_8
    case Q4_0_8_8 => Q4_0_8_8
    case TQ1_0 => TQ1_0
    case TQ2_0 => TQ2_0 
  }



  private val QK_K = 256 // or 64?
  private val QK4_0 = 32
  private val QK8_0 = 32
  
  def this (blockByteSize: Int) = {
    this (blockByteSize, 1, false)
  }

  def this(blockByteSize: Int, elementsPerBlock: Int) ={
    this(blockByteSize, elementsPerBlock, true)
  }

  def this(blockByteSize: Int, elementsPerBlock: Int, isQuantized: Boolean) ={
    
  }

  private def isPowerOf2(n: Int) = n > 0 && (n & (n - 1)) == 0


}