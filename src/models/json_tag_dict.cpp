#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

#include "models/json_tag_dict.h"

// local helpers
static std::shared_ptr<Node> json_to_node(const nlohmann::json &json);

static bool parse_rules(const nlohmann::json &json_rules, Rules &out_rules)
{
    if (!json_rules.is_object())
        return false;

    // unique_categories
    if (json_rules.contains("unique_categories") && json_rules["unique_categories"].is_array())
    {
        out_rules.unique_categories.clear();
        for (const auto &item : json_rules["unique_categories"])
        {
            if (item.is_string())
                out_rules.unique_categories.push_back(item.get<std::string>());
        }
    }

    // exclusive_groups: object of string -> array of arrays
    if (json_rules.contains("exclusive_groups") && json_rules["exclusive_groups"].is_object())
    {
        out_rules.exclusive_groups.clear();
        for (auto it = json_rules["exclusive_groups"].begin(); it != json_rules["exclusive_groups"].end(); ++it)
        {
            const auto &group_arrays = it.value();
            if (!group_arrays.is_array())
                continue;

            std::vector<std::vector<std::string>> category_arrays;
            for (const auto &category_array : group_arrays)
            {
                if (!category_array.is_array())
                    continue;

                std::vector<std::string> categories;
                for (const auto &category : category_array)
                {
                    if (category.is_string())
                        categories.push_back(category.get<std::string>());
                }
                if (!categories.empty())
                    category_arrays.push_back(std::move(categories));
            }
            if (category_arrays.size() >= 2) // Need at least 2 arrays to be exclusive
                out_rules.exclusive_groups[it.key()] = std::move(category_arrays);
        }
    }

    // incompatible_categories: object of category -> array of categories
    if (json_rules.contains("incompatible_categories") && json_rules["incompatible_categories"].is_object())
    {
        out_rules.incompatible_categories.clear();
        for (auto it = json_rules["incompatible_categories"].begin(); it != json_rules["incompatible_categories"].end(); ++it)
        {
            const auto &arr = it.value();
            if (!arr.is_array())
                continue;
            std::vector<std::string> members;
            for (const auto &m : arr)
                if (m.is_string())
                    members.push_back(m.get<std::string>());
            out_rules.incompatible_categories[it.key()] = std::move(members);
        }
    }

    return !out_rules.unique_categories.empty() || !out_rules.exclusive_groups.empty() || !out_rules.incompatible_categories.empty();
}

static CategoryRegistry build_registry_from_rules(const Rules &rules)
{
    CategoryRegistry reg;

    std::set<std::string> all_categories(rules.unique_categories.begin(), rules.unique_categories.end());
    for (const auto &kv : rules.exclusive_groups)
    {
        for (const auto &category_array : kv.second)
        {
            all_categories.insert(category_array.begin(), category_array.end());
        }
    }
    for (const auto &kv : rules.incompatible_categories)
    {
        all_categories.insert(kv.first);
        all_categories.insert(kv.second.begin(), kv.second.end());
    }

    uint8_t bit_index = 0;
    for (const auto &cat : all_categories)
    {
        if (bit_index >= 64)
            break;
        reg.bit_by_category[cat] = bit_index++;
    }

    for (const auto &cat : rules.unique_categories)
    {
        auto it = reg.bit_by_category.find(cat);
        if (it != reg.bit_by_category.end())
            reg.unique_mask_all |= (uint64_t(1) << it->second);
    }

    reg.exclusive_groups.reserve(rules.exclusive_groups.size());
    for (const auto &kv : rules.exclusive_groups)
    {
        CategoryRegistry::ExclusiveGroup eg;
        eg.name = kv.first;
        eg.group_masks.reserve(kv.second.size());

        for (const auto &category_array : kv.second)
        {
            uint64_t mask = 0;
            for (const auto &cat : category_array)
            {
                auto it = reg.bit_by_category.find(cat);
                if (it != reg.bit_by_category.end())
                    mask |= (uint64_t(1) << it->second);
            }
            if (mask != 0) // only add non empty masks
                eg.group_masks.push_back(mask);
        }

        if (eg.group_masks.size() >= 2) // need at least 2 groups to be exclusive
            reg.exclusive_groups.push_back(std::move(eg));
    }

    // build per bit incompatibility masks
    reg.incompat_mask_by_bit.fill(0);
    for (const auto &kv : rules.incompatible_categories)
    {
        auto it_src = reg.bit_by_category.find(kv.first);
        if (it_src == reg.bit_by_category.end())
            continue;
        const uint8_t src_bit = it_src->second;
        uint64_t mask = 0;
        for (const auto &dst : kv.second)
        {
            auto it_dst = reg.bit_by_category.find(dst);
            if (it_dst == reg.bit_by_category.end())
                continue;
            mask |= (uint64_t(1) << it_dst->second);
        }
        reg.incompat_mask_by_bit[src_bit] |= mask;

        // make incompatibility symmetric for convenience
        for (const auto &dst : kv.second)
        {
            auto it_dst = reg.bit_by_category.find(dst);
            if (it_dst == reg.bit_by_category.end())
                continue;
            reg.incompat_mask_by_bit[it_dst->second] |= (uint64_t(1) << src_bit);
        }
    }

    return reg;
}

// factory
std::shared_ptr<Tag_dict> create_tag_dict(const std::string &json_path)
{
    try
    {
        if (!std::filesystem::exists(json_path))
        {
            std::cerr << "JSON file does not exist: " << json_path << std::endl;
            return nullptr;
        }

        std::ifstream file(json_path);
        if (!file.is_open())
        {
            std::cerr << "Failed to open JSON file: " << json_path << std::endl;
            return nullptr;
        }

        nlohmann::json json_data;
        file >> json_data;

        nlohmann::json tags_json;
        if (json_data.is_object() && json_data.contains("tags"))
        {
            tags_json = json_data["tags"];
        }
        else
        {
            // backward compatibility: whole file is the tags tree - legacy
            tags_json = json_data;
        }

        std::shared_ptr<Node> root = json_to_node(tags_json);
        if (!root)
        {
            std::cerr << "Failed to parse tags from JSON" << std::endl;
            return nullptr;
        }

        Rules rules;
        if (json_data.is_object() && json_data.contains("rules"))
        {
            if (!parse_rules(json_data["rules"], rules))
            {
                throw std::runtime_error("Invalid dictionnary rules");
            }
        }
        // else: rules remain empty (valid)

        CategoryRegistry registry = build_registry_from_rules(rules);

        return std::make_shared<Tag_dict>(root, std::move(rules), std::move(registry));
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error creating tag dict from JSON: " << e.what() << std::endl;
        return nullptr;
    }
}

// helper to convert JSON to Node structures
static std::shared_ptr<Node> json_to_node(const nlohmann::json &json)
{
    if (json.is_string())
    {
        return std::make_shared<StringNode>(json.get<std::string>());
    }
    else if (json.is_array())
    {
        std::vector<std::string> values;
        for (const auto &item : json)
        {
            if (item.is_string())
            {
                values.push_back(item.get<std::string>());
            }
        }
        return std::make_shared<ListNode>(values);
    }
    else if (json.is_object())
    {
        std::map<std::string, std::shared_ptr<Node>> children;
        for (auto it = json.begin(); it != json.end(); ++it)
        {
            children[it.key()] = json_to_node(it.value());
        }
        return std::make_shared<MapNode>(children);
    }

    // fallback
    return nullptr;
}