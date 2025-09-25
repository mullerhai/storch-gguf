# Storch-GGUF: A Scala 3 Library for GGUF File Operations

## Overview
Storch-GGUF is a Scala 3 project designed for reading and writing [GGUF](https://github.com/ggerganov/gguf) (Generic GGML Unified Format) files. It enables direct loading of Hugging Face Transformers models and seamless interaction with [storch-llama.cpp](https://github.com/mullerhai/storch-gguf), providing a robust solution for working with large language models in the Scala ecosystem.

## Features
- **GGUF File Handling**: Read and write GGUF files with ease.
- **Hugging Face Integration**: Directly load models from Hugging Face Transformers.
- **storch-llama.cpp Compatibility**: Interact with the torch-llama.cpp framework.

## Getting Started

### Prerequisites
- **Scala 3**: Ensure you have Scala 3 installed on your system.
- **sbt**: Use sbt as the build tool for this project.

### Installation
Add the following dependency to your `build.sbt` file:
```sbt:D:/data/git/storch-gguf/build.sbt
libraryDependencies += "io.github.mullerhai" %% "storch-gguf" % "0.0.3"

```

```  scala 3

import io.github.mullerhai.storchgguf.GGUFReader
import io.github.mullerhai.storchgguf.HuggingFaceLoader

// Load a model from Hugging Face
val model = HuggingFaceLoader.loadModel("gpt2")

// Read the model using GGUFReader
val ggufReader = new GGUFReader(model)
val modelData = ggufReader.read()
```


```` scala 3
import io.github.mullerhai.storchgguf.GGUFWriter

// Assume you have processed model data
val processedData = Map("key" -> "value")

// Write data to a GGUF file
val writer = new GGUFWriter("output.gguf")
writer.write(processedData)
writer.close()
````


```` scala 3
import io.github.mullerhai.storchgguf.StorchLlamaInteractor

// Initialize the interactor
val interactor = new StorchLlamaInteractor()

// Pass the model data to torch-llama.cpp
interactor.processModel(modelData)
````


## Future Plans
We are constantly working to enhance Storch-GGUF. Future plans include:

Support for more large language models and formats.
Improved performance optimizations.
Enhanced error handling and logging.
Contributing
Contributions are welcome! If you'd like to contribute to Storch-GGUF, please follow these steps:

Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes and write tests if applicable.
Submit a pull request.
License
This project is licensed under the Apache License 2.0.

Contact
If you have any questions or suggestions, feel free to open an issue on the GitHub repository.