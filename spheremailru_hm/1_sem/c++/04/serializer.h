#pragma once
#include <iostream>
#include <type_traits>
#include <sstream>

enum class Error {
	NoError,
	CorruptedArchive
};

class Serializer {
	static constexpr char Separator = ' ';

public:
	explicit Serializer(std::ostream& out)
		: out_(out) {}

	template<class T>
	Error save(T& object) {
		return object.serialize(*this);
	}

	template<class... Args>
	Error operator()(Args&&... args) {
		return process(std::forward<Args>(args)...);
	}

private:
	Error save(bool val) {
		out_ << (val ? "true" : "false");
		return Error::NoError;
	}

	Error save(uint64_t val) {
		out_ << val;
		return Error::NoError;
	}

	template<class T, class... Args>
	Error process(T&& val, Args&&... args) {
		Error err_code = save(std::forward<T>(val));
		if (err_code != Error::NoError)
			return err_code;
		out_ << Separator;
		return process(std::forward<Args>(args)...);
	}

	template<class T>
	Error process(T&& val) {
		return save(std::forward<T>(val));
	}

	std::ostream& out_;
};


class Deserializer {

public:
	explicit Deserializer(std::istream& in)
		: in_(in) {}

	template<class T>
	Error load(T &object) {
		return object.serialize(*this);
	}

	template<class... Args>
	Error operator()(Args&&... args) {
		return process(std::forward<Args>(args)...);
	}

private:

	Error load(bool& val) {
		std::string s_val;
		in_ >> s_val;
		if (s_val == "true") {
			val = true;
		}
		else if (s_val == "false") {
			val = false;
		}
		else {
			return Error::CorruptedArchive;
		}
		return Error::NoError;
	}

	Error load(uint64_t& val) {
		std::string s_val;
		in_ >> s_val;
		if (s_val.length() == 0 || s_val[0] == '-')
			return Error::CorruptedArchive;
		size_t pos;
		try {
			val = std::stoull(s_val, &pos);
		}
		catch (std::exception &e) {
			return Error::CorruptedArchive;
		}
		if (pos != s_val.size())
			return Error::CorruptedArchive;
		return Error::NoError;
	}

	template<class T, class... Args>
	Error process(T&& val, Args&&... args) {
		Error err_code = load(std::forward<T>(val));
		if (err_code != Error::NoError)
			return err_code;
		return process(std::forward<Args>(args)...);
	}

	template<class T>
	Error process(T&& val) {
		return load(std::forward<T>(val));
	}

	std::istream& in_;
};

