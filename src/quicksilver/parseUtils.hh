#ifndef PARSE_UTILS_HH
#define PARSE_UTILS_HH

#include <iostream>
#include <string>

class InputBlock;

bool blockStart(const std::string &line, std::string &blockName);
std::string readBlock(InputBlock &block, std::istream &in);

#endif
