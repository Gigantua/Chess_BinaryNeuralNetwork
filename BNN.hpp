#pragma once
#include "LU_REF.hpp";
#include <random>
#include <immintrin.h>
#include <string.h>
#include <atomic>
#include <functional>
#include <bit>
#include <unordered_set>
#include <regex>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <algorithm>
#include <iostream>  
#include <iomanip>   

#include "BNN_Impl.hpp"

namespace BNN
{
	using namespace std;
	//Relevant occupancy masks for Bishop
	static constexpr uint64_t b_mask[64] =
	{
		0x0040201008040200, 0x0000402010080400, 0x0000004020100A00, 0x0000000040221400,
		0x0000000002442800, 0x0000000204085000, 0x0000020408102000, 0x0002040810204000,
		0x0020100804020000, 0x0040201008040000, 0x00004020100A0000, 0x0000004022140000,
		0x0000000244280000, 0x0000020408500000, 0x0002040810200000, 0x0004081020400000,
		0x0010080402000200, 0x0020100804000400, 0x004020100A000A00, 0x0000402214001400,
		0x0000024428002800, 0x0002040850005000, 0x0004081020002000, 0x0008102040004000,
		0x0008040200020400, 0x0010080400040800, 0x0020100A000A1000, 0x0040221400142200,
		0x0002442800284400, 0x0004085000500800, 0x0008102000201000, 0x0010204000402000,
		0x0004020002040800, 0x0008040004081000, 0x00100A000A102000, 0x0022140014224000,
		0x0044280028440200, 0x0008500050080400, 0x0010200020100800, 0x0020400040201000,
		0x0002000204081000, 0x0004000408102000, 0x000A000A10204000, 0x0014001422400000,
		0x0028002844020000, 0x0050005008040200, 0x0020002010080400, 0x0040004020100800,
		0x0000020408102000, 0x0000040810204000, 0x00000A1020400000, 0x0000142240000000,
		0x0000284402000000, 0x0000500804020000, 0x0000201008040200, 0x0000402010080400,
		0x0002040810204000, 0x0004081020400000, 0x000A102040000000, 0x0014224000000000,
		0x0028440200000000, 0x0050080402000000, 0x0020100804020000, 0x0040201008040200
	};

	//Relevant occupancy masks for Rook
	static constexpr uint64_t r_mask[64] =
	{
		0x000101010101017E, 0x000202020202027C, 0x000404040404047A, 0x0008080808080876,
		0x001010101010106E, 0x002020202020205E, 0x004040404040403E, 0x008080808080807E,
		0x0001010101017E00, 0x0002020202027C00, 0x0004040404047A00, 0x0008080808087600,
		0x0010101010106E00, 0x0020202020205E00, 0x0040404040403E00, 0x0080808080807E00,
		0x00010101017E0100, 0x00020202027C0200, 0x00040404047A0400, 0x0008080808760800,
		0x00101010106E1000, 0x00202020205E2000, 0x00404040403E4000, 0x00808080807E8000,
		0x000101017E010100, 0x000202027C020200, 0x000404047A040400, 0x0008080876080800,
		0x001010106E101000, 0x002020205E202000, 0x004040403E404000, 0x008080807E808000,
		0x0001017E01010100, 0x0002027C02020200, 0x0004047A04040400, 0x0008087608080800,
		0x0010106E10101000, 0x0020205E20202000, 0x0040403E40404000, 0x0080807E80808000,
		0x00017E0101010100, 0x00027C0202020200, 0x00047A0404040400, 0x0008760808080800,
		0x00106E1010101000, 0x00205E2020202000, 0x00403E4040404000, 0x00807E8080808000,
		0x007E010101010100, 0x007C020202020200, 0x007A040404040400, 0x0076080808080800,
		0x006E101010101000, 0x005E202020202000, 0x003E404040404000, 0x007E808080808000,
		0x7E01010101010100, 0x7C02020202020200, 0x7A04040404040400, 0x7608080808080800,
		0x6E10101010101000, 0x5E20202020202000, 0x3E40404040404000, 0x7E80808080808000
	};

	//Popcount for 8..64bit types
	template<typename TestType>
	static constexpr unsigned bitcount(TestType x)
	{
		constexpr unsigned BitCount = sizeof(x) * 8;
		static constexpr TestType m1 = static_cast<TestType>(0x5555555555555555ull); //binary: 0101...
		static constexpr TestType m2 = static_cast<TestType>(0x3333333333333333ull); //binary: 00110011..
		static constexpr TestType m4 = static_cast<TestType>(0x0f0f0f0f0f0f0f0full); //binary:  4 zeros,  4 ones ...
		static constexpr TestType m8 = static_cast<TestType>(0x00ff00ff00ff00ffull); //binary:  8 zeros,  8 ones ...
		static constexpr TestType m16 = static_cast<TestType>(0x0000ffff0000ffffull); //binary: 16 zeros, 16 ones ...
		static constexpr TestType m32 = static_cast<TestType>(0x00000000ffffffffull); //binary: 32 zeros, 32 ones
		static constexpr TestType h01 = static_cast<TestType>(0x0101010101010101ull); //the sum of 256 to the power of 0,1,2,3...

		x = (x & m1) +  ((x >> 1) & m1); //put count of each  2 bits into those  2 bits 
		x = (x & m2) +  ((x >> 2) & m2); //put count of each  4 bits into those  4 bits 
		x = (x & m4) +  ((x >> 4) & m4); //put count of each  8 bits into those  8 bits 
		if constexpr (BitCount > 8)  x = (x & m8) +  ((x >> 8) & m8);   //put count of each 16 bits into those 16 bits 
		if constexpr (BitCount > 16) x = (x & m16) + ((x >> 16) & m16); //put count of each 32 bits into those 32 bits 
		if constexpr (BitCount > 32) x = (x & m32) + ((x >> 32) & m32); //put count of each 64 bits into those 64 bits 

		return x;
	}

	const bool PrintDebug = false;

	//Class to solve 64x64 network
	struct BNN {
		struct Rand {
			uint64_t state0;
			uint64_t state1;

			uint64_t next()
			{
				uint64_t s1 = state0;
				const uint64_t s0 = state1;
				state0 = s0;
				s1 ^= s1 << 23;                              // a
				state1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
				return state1 + s0;
			}

			Rand(uint64_t seedx, uint64_t seedy) {
				state0 = seedx;
				state1 = seedy;
			}
		} rand;

		static constexpr int size = 8;

		BNN(uint64_t seedx, uint64_t seedy) : rand(seedx, seedy)
		{
			Randomize();
		}

		uint64_t weights[size];

		void Store(uint64_t storage[size]) {
			memcpy(storage, weights, sizeof(weights));
		}

		void Restore(uint64_t storage[size])
		{
			memcpy(weights, storage, sizeof(weights));
		}


		void Randomize() {
			for (int i = 0; i < size; i++) {
				weights[i] = rand.next();
			}
		}

		void Mutate() {
			//Flip a few bits. Todo performance
			//int bit = rand.next() % 128;
			//int idx = bit / 64;
			//int bitidx = bit % 64;
			//weights[idx] ^= (1ull << bitidx);

			//Flip 4 bits:
			uint64_t random[2] = { rand.next(), rand.next() };  //8 randoms 0...64k
			uint16_t* rands = (uint16_t*)random;
			for (int i = 0; i < 8; i += 2) {
				weights[rands[i] % size] ^= 1ull << (rands[i + 1] % 64);
			}

			//for (int i = 0; i < size; i++) {
			//	weights[i] ^= rand.next() & rand.next() & rand.next() & rand.next() & rand.next();
			//}
		}
		

		int Transform64(uint64_t input) {
			uint8_t target = 0;
			for (int i = 0; i < size; i++) {
				target |= (std::popcount(input ^ weights[i]) < 32) << i;
			}
			return std::popcount(target) < 4;
		}

		int Transform16(uint64_t input) {
			static thread_local uint64_t results[size];
			for (int i = 0; i < size; i++) {
				results[i] = input ^ weights[i];
			}
			using sumover = uint8_t;
			constexpr int sumbits = sizeof(sumover) * 4;
			constexpr int loops = (sizeof(uint64_t) / sizeof(sumover)) * BNN::size;
			constexpr int finalbits = loops / 4;

			sumover* vals = (sumover*)results;
			uint64_t target = 0;

			for (int i = 0; i < loops; i++)
			{
				if (bitcount(vals[i]) < sumbits) {
					target |= (1ull << i);
				}
			}
			return std::popcount(target) > finalbits;
		}
	};


	//Class to solve 16x32 network
	struct BNN_PEXT {
		struct Rand {
			uint64_t state0;
			uint64_t state1;

			uint64_t next()
			{
				uint64_t s1 = state0;
				const uint64_t s0 = state1;
				state0 = s0;
				s1 ^= s1 << 23;                              // a
				state1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
				return state1 + s0;
			}

			Rand(uint64_t seedx, uint64_t seedy) {
				state0 = seedx;
				state1 = seedy;
			}
		} rand;

		static constexpr int size = 32; //2 uint64_t per bit = 

		BNN_PEXT(uint64_t seedx, uint64_t seedy) : rand(seedx, seedy)
		{
			Randomize();
		}

		uint8_t weights[size];

		void Store(uint8_t storage[size]) {
			memcpy(storage, weights, sizeof(weights));
		}

		void Restore(uint8_t storage[size])
		{
			memcpy(weights, storage, sizeof(weights));
		}

		void Randomize() {
			uint64_t* weightsptr = (uint64_t*)weights;
			for (int i = 0; i < size / 8; i++) {
				weightsptr[i] = rand.next();
			}
			for (int i = size - (size % 8); i < size; i++) {
				weights[i] = rand.next();
			}
		}

		void Mutate() {
			//Randomize();
			//return;
			//Flip a few bits. Todo performance
			uint64_t rand64[2] = { rand.next(), rand.next() };
			uint8_t* rand8x16 = (uint8_t*)&rand64;
			const int cnt = 6; //max of 16 bits
			for (int i = 0; i < cnt; i++) {
				weights[rand8x16[i] % size] ^= 1ull << (rand8x16[i + cnt] % 8);
			}
		}

		uint16_t Transform16(uint16_t pext_input) {

			constexpr int Size = size / 2;
			uint8_t* w8 = (uint8_t*)(weights);
			uint8_t* i8 = (uint8_t*)&pext_input;

			uint64_t target = 0;
			for (int i = 0; i < Size * 2; i++)
			{
				target |= (uint64_t)(std::popcount<uint8_t>(i8[i & 1] ^ w8[i]) < 4) << i;
			}
			return std::popcount(target) > Size;
		}
	};

	//Bitmap(uint64_t) to string
	static std::string _map(uint64_t value)
	{
		static std::string str(64 + 7, 'o');
		for (uint64_t i = 0, c = 0; i < 64; i++)
		{
			uint64_t bitmask = (1ull) << i;

			if ((bitmask & value) != 0) str[c++] = 'X';
			else str[c++] = '.';

			if ((i + 1) % 8 == 0 && i != 63) str[c++] = '\n';
		}
		return str;
	}

	struct BNNSample {
		uint64_t mask;
		uint64_t occ;
		uint64_t result;

		uint16_t pextinput;
		uint16_t pextresult;
	};

	static inline int getScore(BNN& network, const std::vector<BNNSample>& samples, const int bitindex) {
		int score = 0;
		const uint64_t bit = 1ull << bitindex;
		for(const auto& sol : samples)
		{
			int bit_res = network.Transform16(sol.occ);
			int bit_ref = (sol.result & bit) >> bitindex;
			if (bit_res != bit_ref) break;
				score++;
		}
		return score;
	}

	static inline int getScorePEXT(BNN_PEXT& network, const std::vector<BNNSample>& samples, const int bitindex) {
		int score = 0;
		const uint16_t bit = 1ull << bitindex;
		for (const auto& sol : samples)
		{
			uint16_t pdedp_network = network.Transform16(sol.pextinput);
			uint16_t bit_ref = (sol.pextresult & bit) >> bitindex;
			if (pdedp_network != bit_ref) break;
				score++;
		}
		return score;
	}

	std::atomic<bool> DoStop = false;
	std::atomic<int> GlobalBest = 0;
	//Creates networks for 1 bit solving a square
	static uint64_t SolveBit(uint64_t bitcount, std::vector<BNNSample>& samples, int square, int* bitsrc, int n) {

		auto rand = std::bind(std::uniform_int_distribution<uint64_t>(), std::default_random_engine{ std::random_device()() });
		const int bitindex = bitsrc[n];

		BNN bnn(rand(), rand());
		uint64_t bestnet[BNN::size];
		bnn.Store(bestnet);

		const uint64_t mask = samples[0].mask;
		const int perfectscore = samples.size();

		int best = -1;
		while (!DoStop) {
			bnn.Restore(bestnet);
			bnn.Mutate();

			int score = getScore(bnn, samples, bitindex);
			if (score >= best) {
				if (score == 0) {
					bnn.Randomize();
				}
				else {
					bnn.Store(bestnet);
				}
				best = score;
				if (best == perfectscore) {
					if (DoStop) return 0;
					DoStop = true;
					if (n >= std::popcount(mask)) std::cout << "Square[" << square << "](0ull << " << 0 << ") =  { ";
					else if (bitindex < 10) std::cout << "Square[" << square << "](1ull << " << bitindex << ")  = { ";
					else std::cout << "Square[" << square << "](1ull << " << bitindex << ") = { ";

					for (int i = 0; i < BNN::size; i++) {
						cout << "0x" << setfill('0') << setw(16) << right << hex << bnn.weights[i] << "ull";
						if (i != (BNN::size-1)) std::cout << ", ";
						else std::cout << "}, \n";
					}
					std::cout << std::dec;

					return 0;
				}
				if (best > GlobalBest)
				{
					GlobalBest = best;
					if (PrintDebug) std::cout << ("\t" + std::to_string(best) + "/" + std::to_string(perfectscore) + "\n");
				}
			}
		}
	}
	std::mutex bestmutex;
	uint8_t globalbest[BNN_PEXT::size];

	//Creates a network that transforms the input into all 14 pdedp values at once!
	static uint64_t SolvePext(uint64_t bitcount, std::vector<BNNSample>& samples, int square, int n) {

		auto rand = std::bind(std::uniform_int_distribution<uint64_t>(), std::default_random_engine{ std::random_device()() });

		BNN_PEXT bnn(rand(), rand());
		uint8_t bestnet[BNN_PEXT::size];

		bnn.Store(bestnet);
		const int perfectscore = samples.size();

		int best = -1;
		int prevscore = 0;
		int tries = 0;
		while (!DoStop) {
			bnn.Mutate();

			int score = getScorePEXT(bnn, samples, n);
			if (score >= best) {
				if (score == 0) {
					bnn.Randomize();
				}
				else {
					//bestmutex.lock();
					//bnn.Store(globalbest);
					//bestmutex.unlock();
					bnn.Store(bestnet);
				}
				best = score;
				if (best == perfectscore) {
					if (DoStop) return 0;
					DoStop = true;
					//if (n >= 14) std::cout << "Square[" << square << "](0ull << " << 0 << ") =  { ";
					//else if (n < 10) std::cout << "Square[" << square << "](1ull << " << n << ")  = { ";
					//else std::cout << "Square[" << square << "](1ull << " << n << ") = { ";
					cout << "\t";
					for (int i = 0; i < BNN_PEXT::size; i++) {

						cout << "0x" << setfill('0') << setw(2) << right << hex << (int)bnn.weights[i];
						if (i != (BNN_PEXT::size - 1)) std::cout << ", ";
						//else std::cout << "}, \n";
						else std::cout << ", \n";
					}
					std::cout << std::dec;

					return 0;
				}
				if (best > GlobalBest)
				{
					GlobalBest = best;
					if (PrintDebug) std::cout << ("\t" + std::to_string(best) + "/" + std::to_string(perfectscore) + "\n");
				}
			}
			else {
				//bnn.Restore(globalbest);
				bnn.Restore(bestnet);
			}
		}
	}

	static double MegaRef() {
		auto rand = std::bind(std::uniform_int_distribution<uint64_t>(), std::default_random_engine{ std::random_device()() });

	}

	static void CompareSolution() {

		auto rnd = std::bind(std::uniform_int_distribution<uint64_t>(), std::default_random_engine{ std::random_device()() });
		int n = 0;
		uint64_t total_ref = 0;
		uint64_t total_new = 0;
		uint64_t volNew = 0;
		uint64_t volRef = 0;
		volatile uint64_t dnOpt = 0;
		while (true) {
			uint64_t occ = rnd(); n++;
			int sq = rnd() % 64;
			auto start = std::chrono::high_resolution_clock::now();
			uint64_t atk = NetworkData::Rook(sq, occ);
			auto t0 = std::chrono::high_resolution_clock::now();
			uint64_t ref = Lookup_Ref::rookAttacks(sq, occ);
			auto end = std::chrono::high_resolution_clock::now();
			volNew ^= atk; volRef ^= ref;

			total_new += std::chrono::duration_cast<std::chrono::nanoseconds>(t0 - start).count();
			total_ref += std::chrono::duration_cast<std::chrono::nanoseconds>(end - t0).count();

			if (atk != ref) std::cout << "ERR";
			if (n == 1000000) {

				double Mperf_new = 1000000 * 1000.0 / total_new;
				double Mperf_ref = 1000000 * 1000.0 / total_ref;
				std::cout << "REF: " << Mperf_ref << "M\n";
				std::cout << "NEW: " << Mperf_new << "M\n\n";
				n = 0; total_ref = 0; total_new = 0; 
				dnOpt = volNew; dnOpt = volRef;
			}
		}
	}

	bool PrintHeaders = false;
	static void FindSolution() {

		for (int square = 0; square < 64; square++) {
			uint64_t mask = r_mask[square];
			uint64_t bitcount = 12; //std::popcount(mask); //14 bits for both rooks and bishops

			uint64_t bits = 1ull << bitcount;
			//std::cout << "\n//Network Config - "<<square<<"\n//" << std::regex_replace(_map(mask), std::regex("\n"), "\n//") << "\n";

			int* indices = new int[bitcount] { };
			{
				//std::cout <<"Mask=" <<mask <<"ull;\n";
				int idx = 0;
				for (int i = 0; i < 64; i++) {
					if (mask & (1ull << i)) {
						indices[idx++] = i;
					}
				}
			}
			std::vector< BNNSample> samples;
			for (uint64_t occ_cfg = 0; occ_cfg < bits; occ_cfg++) {
				uint64_t occ = _pdep_u64(occ_cfg, mask);
				uint64_t result = Lookup_Ref::rookAttacks(square, occ);
				samples.emplace_back(mask, occ, result);
			}

			int threadcnt = std::thread::hardware_concurrency() + 8;
			for (int n = 0; n < bitcount; n++)
			{
				DoStop = false;
				GlobalBest = 0;
				std::vector<std::thread> threads;
				for (int i = 0; i < threadcnt; i++) {
					threads.emplace_back(SolveBit, bitcount, std::ref(samples), square, indices, n);
				}
				for (auto& t : threads) { t.join(); }
			}
			delete[] indices;
		}
	}

//Todo: Manually change strings and masks to train rook, bishop
//#define SOLVE_ROOK
//#define SOLVE_BISHOP

	static void FindSolutionPEXT() {
		const uint64_t* masksource = b_mask;


		std::cout << "struct Maskinfo { \n";
		std::cout << "\tuint64_t Gather;\n";
		std::cout << "\tuint64_t Scatter;\n";
		std::cout << "\tuint32_t Count;\n";
		std::cout << "\tuint32_t Offset;\n};\n";
		std::cout << "static constexpr Maskinfo BishopMasks[] = {\n\t";
		int offset = 0;
		for (int square = 0; square < 64; square++) {
			uint64_t gather  = masksource[square];
			uint64_t scatter = Lookup_Ref::bishopAttacks(square, 0);
			cout << "{ 0x" << setfill('0') << setw(16) << right << hex << gather << "ull, ";
			cout << "0x" << setfill('0') << setw(16) << right << hex << scatter << "ull, ";
			cout << std::dec;
			cout << setfill(' ') << setw(2) << left << std::popcount(scatter) << ", ";
			cout << setfill(' ') << setw(5) << left << offset << " }";
			if (square != 63) cout << ", ";
			if ((square +1) % 8 == 0) cout << "\n\t";

			offset += std::popcount(scatter) * 32;
		}
		std::cout << "};\n\n";
		std::cout << "static constexpr uint8_t BishopWeights[] = {\n";


		for (int square = 0; square < 64; square++) {
			uint64_t mask = masksource[square];
			//uint64_t bitcount = 12;// std::popcount(mask);
			uint64_t bitcount = std::popcount(mask);
			//Mask, Count, uint8_t*[32]

			uint64_t bits = 1ull << bitcount;
			std::vector<BNNSample> samples;
			for (uint64_t occ_cfg = 0; occ_cfg <= bits; occ_cfg++) {
				uint64_t occ = _pdep_u64(occ_cfg, mask);
				uint64_t result = Lookup_Ref::bishopAttacks(square, occ);

				uint64_t mask_lines = Lookup_Ref::bishopAttacks(square, 0);

				BNNSample sample;
				sample.mask = mask;
				sample.occ = occ;
				sample.result = result;
				sample.pextinput  = _pext_u64(occ, mask);
				sample.pextresult = _pext_u64(result, mask_lines); //Speciality: Input is 12 bits - but 14 bits need to be set in the output!
				samples.push_back(sample);

				uint64_t scattered = _pdep_u64(sample.pextresult, mask_lines);
				uint64_t original = sample.result;
				bitcount = std::max(bitcount, (uint64_t)std::popcount(mask_lines)); //12 to 14 bit expansion for weights below
				if (original != scattered)
				{
					std::cout << "IO error!";
				}
			}
			constexpr int threadcnt = 24;
			for (int n = 0; n < bitcount; n++)
			{
				DoStop = false;
				GlobalBest = 0;
				std::vector<std::thread> threads;
				for (int i = 0; i < threadcnt; i++) {
					threads.emplace_back(SolvePext, bitcount, std::ref(samples), square, n);
				}
				for (auto& t : threads) { t.join(); }
			}
		}
		std::cout << "};\n\n";
		std::cout << "All solutions found!" << "\n";
		for (int i = 0; i < 64; i++) {
			for (int n = 0; n < 14; n++) {
				//code print here
			}
			std::cout << "\n";
		}
	}
}

