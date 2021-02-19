//
// Created by ivarh on 10/11/2020.
//

#ifndef BAYESSIAN_SEGMENTATION_CPP_CLIPARSER_H
#define BAYESSIAN_SEGMENTATION_CPP_CLIPARSER_H

#include <string>
#include <list>
#include <vector>
#include <unordered_map>


enum available_options{run};
class Option {

public:
    const std::string &getKey() const;

    void setKey(const std::string &key);

    int getSize() const;

    void setSize(int size);

    const std::vector<std::string> getValue() const;

    void addValue(const std::string &value);

    Option();





    virtual ~Option();

private:
    std::string key;
    int size;
    std::vector<std::string> values;


};

class CLIParser {
private:
    int run_mode = 0;
    std::vector<Option> options;
    static std::unordered_map<std::string, int> get_opts(){
        return {
                {"run", 1},
                {"i",1},
                {"labeldesk",1},
                {"workdir",1},
                {"smooth",1},
                {"shrink",1}
        };
    }

protected:
    static Option parse_option( int* position, char *argv[]);
public:
    virtual ~CLIParser();

    CLIParser();

    void parse_options(int argc, char *argv[]);

    std::vector<std::string> getValue(const std::string& parameter);






};



#endif //BAYESSIAN_SEGMENTATION_CPP_CLIPARSER_H
