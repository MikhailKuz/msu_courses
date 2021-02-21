#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>
#include <algorithm>

int count_thr = 1;
uint64_t size_8mb_in_b = 8388608;
uint64_t size_8mb_in_8b = 1048576;
uint64_t amount_data_gen = size_8mb_in_8b * 5;

enum {
	first_thd = 1,
	second_thd = 2,
	left_side = 0,
	right_side = 1
};

char f_input_name[] = "input.bin";
char f_output_name[] = "output.bin";
char f_copy_pre_s1_name[] = "copy_pre_s1.bin";
char f_copy_pre_s2_name[] = "copy_pre_s2.bin";
char f_tmp11_name[] = "tmp11.bin";
char f_tmp12_name[] = "tmp12.bin";
char f_tmp21_name[] = "tmp21.bin";
char f_tmp22_name[] = "tmp22.bin";
char f_sort_p1_name[] = "sort_p1.bin";
char f_sort_p2_name[] = "sort_p2.bin";

void deleter(FILE* file)
{
	fclose(file);
};

using FILE_ptr = std::unique_ptr<FILE, void(*)(FILE*)>;
using Array_ptr = std::unique_ptr<int64_t[]>;

class Sort_big_file {

	std::mutex m1, m2;
	uint64_t offs1 = 0;
	uint64_t offs2 = 0;
	uint64_t size = 0;

	void thread_sort_f(FILE *f1, FILE *f2) {
		Array_ptr a(new int64_t[size_8mb_in_8b]);
		m1.lock();
		while (!feof(f1)) {
			uint64_t p1 = ftell(f1);
			uint64_t count = fread(a.get(), 8, size_8mb_in_8b, f1);
			m1.unlock();
			std::sort(a.get(), a.get() + count);
			m2.lock();
			uint64_t p2 = ftell(f2);
			fseek(f2, p1, SEEK_SET);
			fwrite(a.get(), 8, count, f2);
			fseek(f2, p2, SEEK_SET);
			m2.unlock();
			m1.lock();
		};
		m1.unlock();
		return;
	}

	void thread_unite_p(int th_num) {
		if (th_num == first_thd) {
			do {
				sort_parts(left_side);
			} while ((offs1 < offs2) && (offs1 <= size));
			remove(f_tmp11_name);
			remove(f_tmp12_name);
			return;
		}
		else {
			do {
				sort_parts(right_side);
			} while (offs1 < offs2);
			remove(f_tmp21_name);
			remove(f_tmp22_name);
			return;
		}
	}

	void sort_parts(int side = left_side) {
		if (side == left_side) {
			uint64_t p1 = offs1 - size_8mb_in_b;
			{
				FILE_ptr f1(fopen(f_input_name, "rb"), deleter);
				if (f1 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f2(fopen(f_tmp11_name, "w+b"), deleter);
				if (f2 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f3(fopen(f_tmp12_name, "w+b"), deleter);
				if (f3 == NULL) {
					std::cerr << "cant open the file";
				}
				fseek(f1.get(), 0, SEEK_SET);
				char c;
				while ((ftell(f1.get()) <= p1) && (!feof(f1.get()))) {
					fread(&c, 1, 1, f1.get());
					fwrite(&c, 1, 1, f2.get());
				}
				while (ftell(f1.get()) <= offs1) {
					fread(&c, 1, 1, f1.get());
					fwrite(&c, 1, 1, f3.get());
				}
			}
			{
				FILE_ptr f1(fopen(f_sort_p1_name, "wb"), deleter);
				if (f1 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f2(fopen(f_tmp11_name, "rb"), deleter);
				if (f2 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f3(fopen(f_tmp12_name, "rb"), deleter);
				if (f3 == NULL) {
					std::cerr << "cant open the file";
				}

				fseek(f1.get(), 0, SEEK_SET);
				fseek(f2.get(), 0, SEEK_SET);
				fseek(f3.get(), 0, SEEK_SET);
				uint64_t tmp;
				while (!feof(f2.get()) || !feof(f3.get())) {
					if (!feof(f2.get()) && !feof(f3.get())) {
						uint64_t tmp2;
						fread(&tmp, 8, 1, f2.get());
						fread(&tmp2, 8, 1, f3.get());
						if (tmp < tmp2) {
							fwrite(&tmp, 8, 1, f1.get());
							fseek(f2.get(), ftell(f2.get()) - 8, SEEK_SET);
						}
						else {
							fwrite(&tmp2, 8, 1, f1.get());
							fseek(f1.get(), ftell(f1.get()) - 8, SEEK_SET);
						}
					}
					else if (feof(f2.get())) {
						fread(&tmp, 8, 1, f3.get());
						fwrite(&tmp, 8, 1, f1.get());
					}
					else {
						fread(&tmp, 8, 1, f2.get());
						fwrite(&tmp, 8, 1, f1.get());
					}
				}
			}
			offs1 += size_8mb_in_b;
		}
		else {
			uint64_t p2 = offs2 + size_8mb_in_b;
			{
				FILE_ptr f1(fopen(f_copy_pre_s2_name, "rb"), deleter);
				if (f1 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f2(fopen(f_tmp21_name, "w+b"), deleter);
				if (f2 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f3(fopen(f_tmp22_name, "w+b"), deleter);
				if (f3 == NULL) {
					std::cerr << "cant open the file";
				}
				fseek(f1.get(), p2, SEEK_SET);
				char c;
				while (!feof(f1.get())) {
					fread(&c, 1, 1, f1.get());
					fwrite(&c, 1, 1, f2.get());
				}
				fseek(f1.get(), offs2, SEEK_SET);
				while (ftell(f1.get()) <= p2) {
					fread(&c, 1, 1, f1.get());
					fwrite(&c, 1, 1, f3.get());
				}
			}
			{
				FILE_ptr f1(fopen(f_sort_p2_name, "wb"), deleter);
				if (f1 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f2(fopen(f_tmp21_name, "rb"), deleter);
				if (f2 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f3(fopen(f_tmp22_name, "rb"), deleter);
				if (f3 == NULL) {
					std::cerr << "cant open the file";
				}

				fseek(f1.get(), offs2, SEEK_SET);
				fseek(f2.get(), 0, SEEK_SET);
				fseek(f3.get(), 0, SEEK_SET);
				uint64_t tmp;
				while (!feof(f2.get()) || !feof(f3.get())) {
					if (!feof(f2.get()) && !feof(f3.get())) {
						uint64_t tmp2;
						fread(&tmp, 8, 1, f2.get());
						fread(&tmp2, 8, 1, f3.get());
						if (tmp < tmp2) {
							fwrite(&tmp, 8, 1, f1.get());
							fseek(f2.get(), ftell(f2.get()) - 8, SEEK_SET);
						}
						else {
							fwrite(&tmp2, 8, 1, f1.get());
							fseek(f1.get(), ftell(f1.get()) - 8, SEEK_SET);
						}
					}
					else if (feof(f2.get())) {
						fread(&tmp, 8, 1, f3.get());
						fwrite(&tmp, 8, 1, f1.get());
					}
					else {
						fread(&tmp, 8, 1, f2.get());
						fwrite(&tmp, 8, 1, f1.get());
					}
				}
			}
			offs2 -= 8388608;
		}
		return;
	}

public:
	void process(int count_thr = 1) {
		if (count_thr == 2) {
			{
				FILE_ptr f1(fopen(f_input_name, "rb"), deleter);
				if (f1 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f2(fopen(f_output_name, "w+b"), deleter);
				if (f2 == NULL) {
					std::cerr << "cant open the file";
				}

				fseek(f1.get(), 0, SEEK_END);
				size = ftell(f1.get());
				if (size <= size_8mb_in_b) {
					int64_t* a = new int64_t[size / 8];
					fseek(f1.get(), 0, SEEK_SET);
					fread(a, 8, size / 8, f1.get());
					std::sort(a, a + size / 8);
					fclose(f1.get());
					fwrite(a, 8, size / 8, f2.get());
					fclose(f2.get());
					delete[] a;
					return;
				}

				fseek(f1.get(), 0, SEEK_SET);

				std::thread t1(&Sort_big_file::thread_sort_f, this, f1.get(), f2.get());
				std::thread t2(&Sort_big_file::thread_sort_f, this, f1.get(), f2.get());
				t1.join();
				t2.join();
			}
			{
				FILE_ptr f1(fopen(f_output_name, "rb"), deleter);
				if (f1 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f2(fopen(f_copy_pre_s1_name, "wb"), deleter);
				if (f2 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f3(fopen(f_copy_pre_s2_name, "wb"), deleter);
				if (f3 == NULL) {
					std::cerr << "cant open the file";
				}

				char c;
				while (!feof(f1.get())) {
					fread(&c, 1, 1, f1.get());
					fwrite(&c, 1, 1, f2.get());
					fwrite(&c, 1, 1, f3.get());
				}
				offs1 = 2 * size_8mb_in_b;
				offs2 = ftell(f1.get()) - 2 * size_8mb_in_b;
			}

			if (size <= 3 * size_8mb_in_b) {
				std::thread t11(&Sort_big_file::thread_unite_p, this, first_thd);
				t11.join();
				{
					FILE_ptr f1(fopen(f_output_name, "wb"), deleter);
					if (f1 == NULL) {
						std::cerr << "cant open the file";
					}
					FILE_ptr f2(fopen(f_copy_pre_s1_name, "rb"), deleter);
					if (f2 == NULL) {
						std::cerr << "cant open the file";
					}

					uint64_t tmp;
					while (!feof(f2.get())) {
						fread(&tmp, 8, 1, f2.get());
						fwrite(&tmp, 8, 1, f1.get());
					}
				}
				remove(f_copy_pre_s1_name);
				remove(f_copy_pre_s1_name);

				return;
			}

			std::thread t11(&Sort_big_file::thread_unite_p, this, first_thd);
			std::thread t21(&Sort_big_file::thread_unite_p, this, second_thd);
			t11.join();
			t21.join();

			// last step: join copy_pre_s1.bin and copy_pre_s2.bin in output.bin
			{
				FILE_ptr f1(fopen(f_output_name, "wb"), deleter);
				if (f1 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f2(fopen(f_copy_pre_s1_name, "rb"), deleter);
				if (f2 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f3(fopen(f_copy_pre_s2_name, "rb"), deleter);
				if (f3 == NULL) {
					std::cerr << "cant open the file";
				}

				uint64_t tmp;
				while (!feof(f2.get()) || !feof(f3.get())) {
					if (!feof(f2.get()) && !feof(f3.get())) {
						uint64_t tmp2;
						fread(&tmp, 8, 1, f2.get());
						fread(&tmp2, 8, 1, f3.get());
						if (tmp < tmp2) {
							fwrite(&tmp, 8, 1, f1.get());
							fseek(f2.get(), ftell(f2.get()) - 8, SEEK_SET);
						}
						else {
							fwrite(&tmp2, 8, 1, f1.get());
							fseek(f1.get(), ftell(f1.get()) - 8, SEEK_SET);
						}
					}
					else if (feof(f2.get())) {
						fread(&tmp, 8, 1, f3.get());
						fwrite(&tmp, 8, 1, f1.get());
					}
					else {
						fread(&tmp, 8, 1, f2.get());
						fwrite(&tmp, 8, 1, f1.get());
					}
				}
			}

			remove(f_copy_pre_s1_name);
			remove(f_copy_pre_s1_name);

			return;
		}
		else if (count_thr == 1) {
			{
				FILE_ptr f1(fopen(f_input_name, "rb"), deleter);
				if (f1 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f2(fopen(f_output_name, "w+b"), deleter);
				if (f2 == NULL) {
					std::cerr << "cant open the file";
				}

				fseek(f1.get(), 0, SEEK_END);
				size = ftell(f1.get());
				if (size <= size_8mb_in_b) {
					int64_t* a = new int64_t[size / 8];
					fseek(f1.get(), 0, SEEK_SET);
					fread(a, 8, size / 8, f1.get());
					std::sort(a, a + size / 8);
					fclose(f1.get());
					fwrite(a, 8, size / 8, f2.get());
					fclose(f2.get());
					delete[] a;
					return;
				}

				fseek(f1.get(), 0, SEEK_SET);

				std::thread t1(&Sort_big_file::thread_sort_f, this, f1.get(), f2.get());
				t1.join();
			}
			{
				FILE_ptr f1(fopen(f_output_name, "rb"), deleter);
				if (f1 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f2(fopen(f_copy_pre_s1_name, "wb"), deleter);
				if (f2 == NULL) {
					std::cerr << "cant open the file";
				}

				char c;
				while (!feof(f1.get())) {
					fread(&c, 1, 1, f1.get());
					fwrite(&c, 1, 1, f2.get());
				}
				offs1 = 2 * size_8mb_in_b;
				offs2 = ftell(f1.get());
			}

			std::thread t11(&Sort_big_file::thread_unite_p, this, first_thd);
			t11.join();
			{
				FILE_ptr f1(fopen(f_output_name, "wb"), deleter);
				if (f1 == NULL) {
					std::cerr << "cant open the file";
				}
				FILE_ptr f2(fopen(f_copy_pre_s1_name, "rb"), deleter);
				if (f2 == NULL) {
					std::cerr << "cant open the file";
				}

				uint64_t tmp;
				while (!feof(f2.get())) {
					fread(&tmp, 8, 1, f2.get());
					fwrite(&tmp, 8, 1, f1.get());
				}
			}
			remove(f_copy_pre_s1_name);

			return;
		}
		else
			std::cout << "Wrong number of threads";
	}
};


void gen_data(uint64_t count) {
	srand(time(0));
	FILE_ptr f1(fopen(f_input_name, "w+b"), deleter);
	if (f1 == NULL) {
		std::cerr << "cant open the file";
	}
	uint64_t tmp = rand();
	for (int i = 1; i <= count; ++i) {
		fwrite(&tmp, 8, 1, f1.get());
		tmp = rand();
	}
}

int main(int argc, char* argv[])
{
	gen_data(amount_data_gen);
	Sort_big_file sbf;
	sbf.process(count_thr);
	return 0;
}
