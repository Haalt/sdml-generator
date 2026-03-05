#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// forward declarations
class Tag_dict;
std::shared_ptr<Tag_dict> create_tag_dict(const std::string &json_path);
std::shared_ptr<Tag_dict> create_tag_dict_int(const std::string &json_path);

struct Node
{
    virtual ~Node() = default;
};

struct StringNode : Node
{
    std::string value;
    StringNode(const std::string &val) : value(val) {}
};

struct ListNode : Node
{
    std::vector<std::string> values;
    ListNode(const std::vector<std::string> &vals) : values(vals) {}
};

struct MapNode : Node
{
    std::map<std::string, std::shared_ptr<Node>> children;
    MapNode(const std::map<std::string, std::shared_ptr<Node>> &ch) : children(ch) {}
};

// rules live alongside the tag tree (not inside it)
struct Rules
{
    std::vector<std::string> unique_categories;
    std::vector<std::string> filter_categories;
    bool has_filter_categories{false};
    std::unordered_map<std::string, std::vector<std::vector<std::string>>> exclusive_groups; // group name -> array of category arrays (mutually exclusive arrays)
    std::unordered_map<std::string, std::vector<std::string>> incompatible_categories;       // category -> categories it cannot co-exist with
};

/**
 * Runtime registry mapping category names to bit positions and precomputed masks
 * Uses a 64 bit mask
 */
struct CategoryRegistry
{
    std::unordered_map<std::string, uint8_t> bit_by_category; // category -> bit index [0..63]
    uint64_t unique_mask_all = 0;                             // mask of all unique categories

    struct ExclusiveGroup
    {
        std::string name;
        std::vector<uint64_t> group_masks; // each mask represents one category array in the group
    };

    std::vector<ExclusiveGroup> exclusive_groups;

    // fast conflict checks: per bit incompatible categories mask
    std::array<uint64_t, 64> incompat_mask_by_bit{}; // for bits not used, entry is 0
};

class Tag_dict
{
    inline static std::shared_ptr<Tag_dict> s_instance_ = nullptr; // c++17 inline definition

    std::shared_ptr<Node> rootNode; // tags root only
    Rules rules_;
    CategoryRegistry registry_;

public:
    // Default constructor for creating uninitialized instances
    Tag_dict() = default;

    explicit Tag_dict(std::shared_ptr<Node> root) : rootNode(std::move(root)) {}
    Tag_dict(std::shared_ptr<Node> root, Rules rules, CategoryRegistry registry)
        : rootNode(std::move(root)), rules_(std::move(rules)), registry_(std::move(registry)) {}

    /**
     * @brief Loads tag dictionary from JSON into this instance (non-static)
     * @param path Path to the JSON file
     * @return true if loading succeeded
     */
    bool loadFromJsonInstance(const std::string &path)
    {
        std::shared_ptr<Tag_dict> tg = create_tag_dict(path);
        if (tg == nullptr)
            return false;
        rootNode = std::move(tg->rootNode);
        rules_ = std::move(tg->rules_);
        registry_ = std::move(tg->registry_);
        return true;
    }

    static bool loadFromJson(const std::string &path)
    {
        if (s_instance_ != nullptr)
        {
            std::cerr << "Tag dict already initialized" << std::endl;
            return true;
        }
        std::shared_ptr<Tag_dict> tg = create_tag_dict(path);
        if (tg == nullptr)
            return false;
        s_instance_ = std::move(tg);
        return true;
    }

    static bool loadFromIntJson(const std::string &path)
    {
        if (s_instance_ != nullptr)
        {
            std::cerr << "Tag dict already initialized" << std::endl;
            return true;
        }
        std::shared_ptr<Tag_dict> tg = create_tag_dict_int(path);
        if (tg == nullptr)
            return false;
        s_instance_ = std::move(tg);
        return true;
    }

    static const std::shared_ptr<const Tag_dict> getInstance(void)
    {
        return s_instance_;
    }

    std::shared_ptr<Node> getNode(const std::string &path) const
    {
        if (path.empty())
            return rootNode;

        std::istringstream ss(path);
        std::string token;
        std::shared_ptr<Node> currentNode = rootNode;

        while (std::getline(ss, token, '.'))
        {
            auto mapNode = std::dynamic_pointer_cast<MapNode>(currentNode);
            if (mapNode && mapNode->children.count(token))
                currentNode = mapNode->children[token];
            else
                return nullptr; // path does not exist
        }
        return currentNode;
    }

    std::vector<std::string> getListNode(const std::string &path) const
    {
        auto node = getNode(path);
        if (!node)
            return {};
        if (auto listNode = std::dynamic_pointer_cast<ListNode>(node))
            return listNode->values;
        if (auto stringNode = std::dynamic_pointer_cast<StringNode>(node))
            return {stringNode->value};
        if (std::dynamic_pointer_cast<MapNode>(node))
        {
            std::vector<std::string> result;
            flattenNode(node, result);
            return result;
        }
        return {};
    }

    void flattenNode(const std::shared_ptr<Node> &node, std::vector<std::string> &result) const
    {
        if (auto stringNode = std::dynamic_pointer_cast<StringNode>(node))
        {
            result.push_back(stringNode->value);
        }
        else if (auto listNode = std::dynamic_pointer_cast<ListNode>(node))
        {
            result.insert(result.end(), listNode->values.begin(), listNode->values.end());
        }
        else if (auto mapNode = std::dynamic_pointer_cast<MapNode>(node))
        {
            for (const auto &[key, childNode] : mapNode->children)
            {
                flattenNode(childNode, result);
            }
        }
    }

    std::vector<std::string> flatten() const
    {
        std::vector<std::string> result;
        flattenNode(rootNode, result);
        return result;
    }

    // rules accessors
    const Rules &getRules() const { return rules_; }
    const CategoryRegistry &getRegistry() const { return registry_; }
};
