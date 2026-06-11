# Experiment Workflow

## Experimental Assumptions

- The length of the original message is known.
- No hash input hardening values such as salt or pepper are applied.

## 1. Data Generation

### 1.1 Message Generation
Generate original messages to be used for training the Diffusion Model.  
Messages can be generated either at the bit level or at the character level. In character mode, the candidate character set is specified separately.

**Input**

- Bit length of the original message
- Generation mode: bit or character
- Candidate character set: used in character mode
- Number of samples to generate

**Output**

- Original message
- Message length metadata

### 1.2 Hash Computation and Log Collection
Run the selected Hash Algorithm on each generated original message.  
In addition to the final hash value, intermediate computation logs that may be used for Diffusion Model training are also stored.

**Input**

- Original message
- Hash Algorithm type
- Hash configuration: word size, block size, byte order, etc.

**Output**

- Final hash value
- Intermediate Hash computation logs
- Metadata that links each message to its hash value

Intermediate computation logs are stored in JSON format.  
These logs can later be encoded as images or matrices and used as training data for the Diffusion Model.

## 2. Image Encoding

### 2.1 RGB Encoding
Convert the parts of the Hash computation logs or message byte sequence selected for training into RGB images.  
The detailed RGB encoding and decoding rules are defined in [Encoding.md](./Encoding.md).

**Encoding**

```text
Byte sequence -> RGB value sequence -> PNG image
```

**Decoding**

```text
PNG image -> RGB value sequence -> Byte sequence
```

RGB encoding maps byte values to specific regions in RGB space.  
After the Diffusion Model generates an image, RGB values are extracted from the image and decoded back into the original byte sequence using the same rules.

### 2.2 Matrix Encoding
Convert byte-level or numeric data into an image with a matrix structure.  
Matrix encoding uses spatial patterns as the learning target rather than raw RGB values.

**Encoding**

```text
Byte sequence -> Matrix representation -> PNG image
```

**Decoding**

```text
PNG image -> Matrix representation -> Byte sequence
```

The matrix structure is generated according to the rules defined in [Encoding.md](./Encoding.md).  
During inference, the matrix structure is reconstructed from the generated image, and the recovered pattern is decoded back into byte values.

## 3. Training Dataset Construction
Each training sample for the Diffusion Model is composed of an image and its condition information.

**Image Data**

- PNG file encoded from the original message or Hash computation logs
- RGB-encoded image or Matrix-encoded image

**Condition Information**

- Hash value of the original message
- Bit length of the original message
- Hash Algorithm type
- Hash computation step or log position, if needed

The dataset must manage image files and metadata together.  
It should be possible to trace which original message, hash value, and bit length correspond to each image.

## 4. Text-Based Diffusion Model Workflow

### 4.1 Overview
The Text-Based Diffusion Model adopts a text-to-image generation structure.  
The model receives the hash value and message length as conditions and learns to generate the encoded image corresponding to those conditions.

The training target image is a PNG file encoded from the original message or Hash computation logs.  
The condition information consists of the hash value and message length linked to that image.

### 4.2 Training Pipeline
During training, the Forward Process is applied to the encoded image, and the Reverse Process is learned using the condition information.

1. Encode the original message or Hash computation logs as a PNG image.
2. Add Gaussian noise to the PNG image at each timestep.
3. Provide the hash value and message length as conditions.
4. Train a U-Net or denoising network to predict the added noise.
5. Minimize the loss between the actual noise and the predicted noise.

Through this process, the model learns which image distribution is associated with a given condition.  
After training, the model can generate an encoded image that matches the same condition.

### 4.3 Inference Pipeline
During inference, the hash value and message length are used as conditions to generate an image.

1. Convert the input hash value into a condition embedding.
2. Provide the bit length of the original message as an additional condition.
3. Start reverse sampling from pure Gaussian noise.
4. Generate an encoded image using the trained denoising network.
5. Decode the generated image into a byte sequence using the RGB or Matrix decoding rules.
6. Use the recovered byte sequence as a candidate message.

Because generated results are affected by stochastic sampling, multiple candidates may be produced from the same condition.  
Therefore, inference results should be managed as a candidate set rather than a single output.

## 5. Process-Based Diffusion Model Workflow

### 5.1 Overview
The Process-Based Diffusion Model uses the intermediate computation process of the Hash Algorithm as the learning target instead of relying only on the final hash value.  
It analyzes the structure in which the internal state of the Hash computation is updated step by step by mapping it to the timestep flow of a Diffusion Model.

### 5.2 Training Direction
Intermediate Hash Algorithm states, round-level outputs, and block-level update logs are converted into a sequence or image representation.  
The model learns structural patterns in the computation process from this representation.

**Candidate Training Data**

- Round-level internal states
- Block-level compression results
- Word schedule or message schedule
- Relationship between the final hash value and intermediate states

### 5.3 Inference Direction
During inference, the final hash value or partial intermediate states are provided as conditions to generate possible computation processes or original representation candidates.  
The generated candidates are then decoded back into byte sequences.

The Process-Based approach requires a more complex training dataset than the Text-Based approach, but it can make greater use of the internal structure of the Hash Algorithm.

## 6. Result Verification
The outputs generated by the Diffusion Model are ultimately verified by applying the Hash Algorithm again.

1. Decode the generated image into a byte sequence.
2. Interpret the decoded byte sequence as a candidate message.
3. Apply the same Hash Algorithm to the candidate message.
4. Compare the computed result with the hash value used as the generation condition.
5. If the values match, record the result as a successful candidate.

Verification results are stored together with the candidate message, generated image, condition, hash value, and decoding success status.  
This information is used for model evaluation and follow-up experiment analysis.
