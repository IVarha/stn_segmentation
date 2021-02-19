//
// Created by ivarh on 10/11/2020.
//

#include <iostream>
#include "CLIParser.h"

CLIParser::CLIParser() {
    CLIParser::options = std::vector<Option>();
}

void CLIParser::parse_options(int argc, char **argv) {
    int i = 1;
    while (i < argc){
        int* ii = new int(i);
        std::cout << "Hello, World!" << std::endl;
        Option option = parse_option(ii,argv);
        CLIParser::options.push_back(option);

        //rewrite i
        i = *ii;
        i++;
    }
}

Option CLIParser::parse_option(int *position, char **argv) {

    int pos = *position;

    // result
    Option result = Option();

    // parse
    std::string tmp_command = std::string(argv[pos]);
    if (tmp_command[0]== '-'){
        tmp_command = tmp_command.substr(1);
    };
    //opts
    std::unordered_map<std::string,int> opts = CLIParser::get_opts();
    std::unordered_map<std::string,int>::const_iterator el = opts.find(tmp_command);
    if ( el == opts.end() )
        throw "Option not found";

    result.setKey(el->first);
    result.setSize(el->second);
    for (int cnt = pos + 1; cnt <= pos + el->second; cnt++){
        std::string val = std::string(argv[cnt]);
        result.addValue(val);
    }

    *position = pos + el->second;
    std::cout << result.getSize();
    return result;
}

CLIParser::~CLIParser() {
    CLIParser::options.clear();
}

std::vector<std::string> CLIParser::getValue(const std::string& parameter) {
    for (Option const& value: options)
    {
        const std::string& key_t = value.getKey();
        if (key_t == parameter){
            return value.getValue();
        }

    }
    return std::vector<std::string>();
}


Option::Option() {
    values = std::vector<std::string>();

}

Option::~Option() {
    values.clear();
}

const std::string &Option::getKey() const {
    return key;
}

void Option::setKey(const std::string &key) {
    Option::key = key;
}

int Option::getSize() const {
    return size;
}

void Option::setSize(int size) {
    Option::size = size;
}

const std::vector<std::string> Option::getValue() const {
    return Option::values;
}



void Option::addValue(const std::string &value) {
    if (Option::size > Option::values.size()){
        Option::values.push_back(value);
    } else{
        throw "Max size exceeded for option" + Option::key;
    }
}
