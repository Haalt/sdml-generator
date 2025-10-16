#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "models/tag_dict.h"

// dactory to create the appropriate tag dictionary (tags + rules)
std::shared_ptr<Tag_dict> create_tag_dict(const std::string &json_path);

// create a JSON representation of a Node tree (tags only)
nlohmann::json node_to_json(const std::shared_ptr<Node> &node);
