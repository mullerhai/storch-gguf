import java.nio.file.Paths

import torch.gguf.GGUF

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
@main
def main(): Unit =
  // TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
  // to see how IntelliJ IDEA suggests fixing it.
  (1 to 5).map(println)
  val path = "Qwen3-Embedding-0.6B-Q8_0.gguf"
  //  val path = "D:\\data\\git\\storch-image\\Qwen3-Embedding-0.6B-Q8_0.gguf"
  val file = Paths.get(path)
  val gguf = GGUF.read(file)
  println(gguf.getVersion)
  println(gguf.getTensors)
  for (i <- 1 to 5) do
  //TIP Press <shortcut actionId="Debug"/> to start debugging your code. We have set one <icon src="AllIcons.Debugger.Db_set_breakpoint"/> breakpoint
  // for you, but you can always add more by pressing <shortcut actionId="ToggleLineBreakpoint"/>.
  println(s"i = $i")
