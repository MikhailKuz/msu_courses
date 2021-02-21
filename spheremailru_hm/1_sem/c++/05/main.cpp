#include <iostream>
#include <condition_variable>
#include <thread>

int count = 500000;

bool flag = true;

std::mutex m;
std::condition_variable dataReady;

void th2() // thread 1
{
	for (int i = 0; i < count; i++) {
		std::unique_lock<std::mutex> lock(m);
		dataReady.wait(lock, []() {return !flag; });
		std::cout << "pong\n";
		flag = true;
		dataReady.notify_one();
	}
}

void th1() // thread 2
{
	for (int j = 0; j < count; j++) {
		std::unique_lock<std::mutex> lock(m);
		dataReady.wait(lock, []() {return flag; });
		std::cout << "ping\n";
		flag = false;
		dataReady.notify_one();
	}
}

int main()
{
	std::thread t1(th1);
	std::thread t2(th2);
	t1.join();
	t2.join();
	
	system("pause");
	return 0;
}