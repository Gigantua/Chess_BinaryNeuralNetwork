# Chess_BinaryNeuralNetwork
## Training and Code Emitting Library for Binary Neural Networks

Sample code can infer 52Million Queen Moves/s on a random board. 
Implemented layouts are 64x64x1
or 16x32x1

### Getting Started

Copy paste the prduced weights as input for this inference code. 
A Binary Neural Network works like a normal neuronal network with the speciality that inputs can be single bits - 
and the dotproduct is defined as a⋅b = popcount(a^b)

And the activation function is if more than half the bits of a⋅b are set (relu). For a reduction of 32 input bits the resulting code is this:
popcount(a^b) > 16

In literature these networks are called XNor networks. These have the same mathematical structure as a normal matrix multiplication + activationfunction. 
> **_NOTE:_** Inversion of xor is not needed when the sign comparison in Relu is inversed. Thus the computational efficiency is very good!
---
**This network code infers 14 networks after each other to produce all 14 possible attacked bits in a 64 bit chessboard starting from any location:**

```
static uint64_t Vector32_Unrolled(int sq, uint64_t occ, uint64_t gather, uint64_t scatter, uint32_t count, const uint8_t* weights) {
	const __m256i input = _mm256_set1_epi16(_pext_u64(occ, gather));
	uint32_t result = 0;

	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 0  * 32)))))) > 16) << 0 ;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 1  * 32)))))) > 16) << 1 ;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 2  * 32)))))) > 16) << 2 ;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 3  * 32)))))) > 16) << 3 ;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 4  * 32)))))) > 16) << 4 ;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 5  * 32)))))) > 16) << 5 ;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 6  * 32)))))) > 16) << 6 ;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 7  * 32)))))) > 16) << 7 ;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 8  * 32)))))) > 16) << 8 ;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 9  * 32)))))) > 16) << 9 ;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 10 * 32)))))) > 16) << 10;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 11 * 32)))))) > 16) << 11;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 12 * 32)))))) > 16) << 12;
	result |= (std::popcount<uint32_t>(_mm256_movemask_epi8(ChessBNN::popcount8x32_SmallerThan4(_mm256_xor_si256(input, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + 13 * 32)))))) > 16) << 13;
	
	return _pdep_u64(result, scatter);
}

```
