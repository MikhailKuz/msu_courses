#include <cstdlib>
#include <iostream>
#include <memory>
#include "numbers.dat"

bool is_prime(int);
int kol_vo(int, int, const int*);

int main(int argc, char* argv[])
{
	if ((argc == 1) || (argc % 2 != 1))
		return -1;
	int count = 0;
	for (int i = 1; i < argc; i += 2) {
		int el_0 = atoi(argv[i]);
		int el_1 = atoi(argv[i + 1]);
		if ((el_0 < Data[0]) || (el_1 < Data[0]) ||
			(el_0 > Data[Size - 1]) || (el_1 > Data[Size - 1])) {
			std::cout << 0;
			continue;
		}
		int i1 = -1;
		int i2 = -1;
		int j = 0;
		while (j < Size) {
			if ((i1 == -1) && (Data[j] == el_0))
				i1 = j;
			if (Data[j] == el_1)
				i2 = j;
			if ((i1 != -1) && (i2 != -1))
				break;
			++j;
		}
		if ((i1 == -1) || (i2 == -1) || (i1 > i2)) {
			std::cout << 0;
			continue;
		}
		count += kol_vo(i1, i2, Data);
		std::cout << count << std::endl;
		count = 0;
	}
	return 0;
}

bool is_prime(int number) {
	if (number <= 1)
		return false;
	for (int i = 2; i*i <= number; ++i)
		if (number % i == 0)
			return false;
	return true;
}

int kol_vo(int i1, int i2, const int* Data) {
	int count = 0;
	for (int j = i1; j <= i2; ++j) {
		if (is_prime(Data[j]) == true)
			++count;
	}
	return count;
}