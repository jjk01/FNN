#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <iostream>
#include <string>
#include <vector>

class Exception{
    std::vector<std::string> msg;
public:
    Exception() = default;
  Exception(const std::string arg){msg.push_back(arg);}
  Exception(const std::string arg, const Exception & e): msg(e.msg){msg.push_back(arg);}
  void what(){
      for (auto m = msg.begin(); m != msg.end(); m++){
          std::cout << *m << "\n" << " ";
      }
  }
};


#endif /* EXCEPTION_HPP */
