#include <iostream> // I/O
#include <cctype>
#include <sstream>
using namespace std;

enum class Kind : char {
	number, end,
	plus = '+', minus = '-', mul = '*', div = '/'
};

struct Token {
	Kind kind;
	int64_t	number_value;
};

class Token_stream {
public:
	Token_stream(istream& s) : ip{ &s }, owns{ false } { }
	Token_stream(istream* p) : ip{ p }, owns{ true } { }
	~Token_stream() { close(); }
	Token get(); // read and return next token
	Token& current() { return ct; }; // most recently read token
	void set_input(istream& s) { close(); ip = &s; owns = false; }
	void set_input(istream* p) { close(); ip = p; owns = true; }
private:
	int errors = 0;
	void close() { if (owns) delete ip; }
	istream* ip; // pointer to an input stream
	bool owns; // does the Token_stream own the istream?
	Token ct{ Kind::end }; // current token
};

Token Token_stream::get()
{
	char ch = 0;
	do { // skip whitespace except ’\n’
		if (!ip->get(ch)) return ct = { Kind::end };
	} while (ch != '\n' && isspace(ch));

	switch (ch) {
	case 0:
		return ct = { Kind::end }; // assign and return
	case '*':
	case '/':
	case '+':
	case '-':
		return ct = { static_cast<Kind>(ch) };
	case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
	case '.':
		ip->putback(ch); // put the first digit (or .) back into the input stream
		*ip >> ct.number_value; // read number into ct
		ct.kind = Kind::number;
		return ct;
	default: // name, name =, or error
		this->errors += 1;
		return ct = { Kind::end };
	}
}

class Calculator {

public:
	Calculator(Token_stream t) : ts{ t }, errors{ 0 }, result{ 0 } { }
	Calculator() : ts{ cin }, errors{ 0 }, result{ 0 } {}

	int64_t expr(bool get) // add and subtract
	{
		int64_t left = term(get);
		for (;;) { // ‘‘forever’’
			switch (ts.current().kind) {
			case Kind::plus:
				left += term(true);
				break;
			case Kind::minus:
				left -= term(true);
				break;
			default:
				return left;
			}
		}
	}

	int64_t term(bool get) // multiply and divide
	{
		int64_t left = prim(get);
		for (;;) {
			switch (ts.current().kind) {
			case Kind::mul:
				left *= prim(true);
				break;
			case Kind::div:
				if (auto d = prim(true)) {
					left /= d;
					break;
				}
				errors += 1;
				return 0;
			default:
				return left;
			}
		}
	}

	int64_t prim(bool get) // handle primaries
	{
		if (get) ts.get(); // read next token
		switch (ts.current().kind) {
		case Kind::number: // floating-point constant
		{
			int64_t v = ts.current().number_value;
			ts.get();
			return v;
		}
		case Kind::minus: // unar y minus
			return -prim(true);
		default:
			errors += 1;
			return 0;
		}
	}

	int get_errors() {
		return errors;
	}

	int get_result() {
		return result;
	}

	void calculate()
	{
		ts.get();
		if (ts.current().kind == Kind::end) return;
		result = expr(false);
		return;
	}

	Token_stream ts;
private:
	int64_t result;
	int errors;
};



int main(int argc, char* argv[])
{
	Token_stream ts{ cin };
	Calculator cl = Calculator(ts);
	switch (argc) {
	case 1:
		cout << "error";
		return 1;
	case 2: // read from argument string
		cl.ts.set_input(new istringstream{ argv[1] });
		break;
	default:
		cout << "error";
		return 1;
	}
	cl.calculate();
	if (cl.get_errors() != 0) 
		cout << cl.get_result();
	else{
		cout << "error";
		return 1;
	}
		cout << cl.get_result();
	return 0;
}
