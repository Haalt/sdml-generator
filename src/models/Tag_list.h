#pragma once

#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "models/tag_dict.h"
#include "utils/fast_rand.hpp"

// Forward declaration
namespace generator
{
    class Tokenizer;
}

#include "tokenizer/tokenizer.hpp"

// struct to store tag information
struct TagInfo
{
    std::string tag;
    std::vector<int64_t> precomputed_tokens; // precomputed token IDs
    uint64_t categories_mask;                // bitmask of categories this tag belongs to (from CategoryRegistry)
};

class Tag_list
{
private:
    // c++17 inline definitions
    inline static std::vector<TagInfo> s_allowed_tags;
    inline static std::once_flag initFlag_;
    inline static std::unordered_map<std::string, size_t> s_index_by_tag; // tag -> index in s_allowed_tags

    uint64_t m_allowed_unique_mask = 0;      // Allowed unique category bits remaining
    uint64_t m_selected_categories_mask = 0; // All categories already selected (for incompatibility checks)

    // exclusive groups state: track which group array is active for each named group
    std::unordered_map<std::string, int> m_active_exclusive_groups; // group_name -> active_array_index (-1 = None)

    std::vector<std::string> m_tags;
    std::vector<std::vector<int64_t>> m_precomputed_tokens; // parallel to m_tags
    std::unordered_set<std::string> m_tags_set;             // for quick lookup

    // helper method to check if a tag's categories conflict with exclusive groups
    bool checkExclusiveGroupsConflict(const uint64_t tag_categories, const CategoryRegistry &reg) const
    {
        for (const auto &eg : reg.exclusive_groups)
        {
            const auto &active_it = m_active_exclusive_groups.find(eg.name);
            if (active_it == m_active_exclusive_groups.end())
                continue;

            const int active_array_index = active_it->second;

            // find which arrays this tag belongs to
            int tag_array_index = -1;
            for (size_t i = 0; i < eg.group_masks.size(); ++i)
            {
                if ((tag_categories & eg.group_masks[i]) != 0)
                {
                    tag_array_index = static_cast<int>(i);
                    break;
                }
            }

            // if tag belongs to an array in this exclusive group
            if (tag_array_index != -1)
            {
                // if no array is active yet, this tag can set the active array
                if (active_array_index == -1)
                    continue;

                // if a different array is already active, conflict
                if (active_array_index != tag_array_index)
                    return true;
            }
        }
        return false;
    }

    // update exclusive groups state when adding a tag
    void updateExclusiveGroupsState(const uint64_t tag_categories, const CategoryRegistry &reg)
    {
        for (const auto &eg : reg.exclusive_groups)
        {
            auto &active_array_index = m_active_exclusive_groups[eg.name];

            // if no array is active, check if this tag activate one
            if (active_array_index == -1)
            {
                for (size_t i = 0; i < eg.group_masks.size(); ++i)
                {
                    if ((tag_categories & eg.group_masks[i]) != 0)
                    {
                        active_array_index = static_cast<int>(i);
                        break;
                    }
                }
            }
        }
    }

    static void initialize_allowed_tags(const generator::Tokenizer *tokenizer = nullptr)
    {
        std::call_once(initFlag_, [tokenizer]()
                       {
    if (!s_allowed_tags.empty())
      return; // already initialized

    // oprecompute tag information
    std::unordered_map<std::string, TagInfo> tag_info_map;
    const std::shared_ptr<const Tag_dict> tg = Tag_dict::getInstance();
    std::vector<std::string> all_tags = tg->flatten();

    // Initialize TagInfo for all tags
    for (const auto &tag : all_tags)
    {
      TagInfo info;
      info.tag = tag;
      info.categories_mask = 0;
      if (tokenizer) {
        info.precomputed_tokens = tokenizer->precomputeTagTokens(tag);
      }
      tag_info_map[tag] = std::move(info);
    }

    // map categories (from registry) to tag masks by reading the list at each category path
    const CategoryRegistry &reg = tg->getRegistry();
    for (const auto &kv : reg.bit_by_category)
    {
      const std::string &category_path = kv.first;
      const uint8_t bit_index = kv.second;
      uint64_t bit = (uint64_t(1) << bit_index);

      std::vector<std::string> tags = tg->getListNode(category_path);
      for (const auto &tag : tags)
      {
        tag_info_map[tag].categories_mask |= bit;
      }
    }

    // build s_allowed_tags from tag_info_map
    s_allowed_tags.reserve(tag_info_map.size());
    s_index_by_tag.reserve(tag_info_map.size());
    for (const auto &pair : tag_info_map)
    {
      s_index_by_tag.emplace(pair.first, s_allowed_tags.size());
      s_allowed_tags.push_back(pair.second);
    } });
    }

public:
    void reset()
    {
        const std::shared_ptr<const Tag_dict> tg = Tag_dict::getInstance();
        const CategoryRegistry &reg = tg->getRegistry();

        m_allowed_unique_mask = reg.unique_mask_all; // All unique categories allowed at start
        m_selected_categories_mask = 0;

        // initialize exclusive groups state, all groups start with no active array
        m_active_exclusive_groups.clear();
        for (const auto &eg : reg.exclusive_groups)
        {
            m_active_exclusive_groups[eg.name] = -1; // -1: no array is active
        }

        m_tags.clear();
        m_precomputed_tokens.clear();
        m_tags_set.clear();
    }

    Tag_list(const generator::Tokenizer *tokenizer = nullptr)
    {
        m_tags.reserve(100);
        m_precomputed_tokens.reserve(100);

        initialize_allowed_tags(tokenizer);

        reset();
    }

    const std::vector<std::string> &get_tags(void) const { return m_tags; }
    const std::vector<std::vector<int64_t>> &get_precomputed_tokens(void) const { return m_precomputed_tokens; }

    void set_tags(const std::vector<std::string> &tags)
    {
        reset(); // start by reseting the state

        const std::shared_ptr<const Tag_dict> tg = Tag_dict::getInstance();
        const CategoryRegistry &reg = tg->getRegistry();

        m_tags = tags;
        m_tags_set.clear();
        m_tags_set.reserve(tags.size() * 2);
        m_tags_set.insert(tags.begin(), tags.end());

        // rebuild precomputed tokens from tags
        m_precomputed_tokens.clear();
        m_precomputed_tokens.reserve(tags.size());
        for (const auto &tag : tags)
        {
            auto it = s_index_by_tag.find(tag);
            if (it != s_index_by_tag.end())
            {
                const TagInfo &info = s_allowed_tags[it->second];
                m_precomputed_tokens.push_back(info.precomputed_tokens);
            }
        }

        // rebuild state from the provided tags
        for (const auto &tag : tags)
        {
            auto it = s_index_by_tag.find(tag);
            if (it != s_index_by_tag.end())
            {
                const TagInfo &info = s_allowed_tags[it->second];
                const uint64_t tag_categories = info.categories_mask;
                const uint64_t tag_unique = (tag_categories & reg.unique_mask_all);

                if (tag_unique)
                    m_allowed_unique_mask &= ~tag_unique;
                m_selected_categories_mask |= tag_categories;
                updateExclusiveGroupsState(tag_categories, reg);
            }
        }
    }

    void insert_tags(const std::vector<std::string> &tags)
    {
        m_tags.insert(m_tags.end(), tags.begin(), tags.end());

        // add precomputed tokens for the new tags
        for (const auto &tag : tags)
        {
            auto it = s_index_by_tag.find(tag);
            if (it != s_index_by_tag.end())
            {
                const TagInfo &info = s_allowed_tags[it->second];
                m_precomputed_tokens.push_back(info.precomputed_tokens);
            }
        }
        // m_tags_set.insert(tags.begin(), tags.end());
    }

    void generate_tags(const size_t num = 10)
    {
        const std::shared_ptr<const Tag_dict> tg = Tag_dict::getInstance();
        const CategoryRegistry &reg = tg->getRegistry();

        size_t max_attempts = s_allowed_tags.size() * 2; // prevent infinite loops xd
        size_t attempts = 0;

        while (m_tags.size() < num && attempts < max_attempts)
        {
            ++attempts;

            // const TagInfo &rand_tag_info = s_allowed_tags[rand() % s_allowed_tags.size()];
            const TagInfo &rand_tag_info = s_allowed_tags[rng::irange(0, s_allowed_tags.size() - 1)];
            const std::string &rand_tag = rand_tag_info.tag;

            if (m_tags_set.count(rand_tag))
                continue;

            const uint64_t tag_categories = rand_tag_info.categories_mask;

            // force uniqueness: if tag has any unique categories, they must still be allowed
            const uint64_t tag_unique = (tag_categories & reg.unique_mask_all);
            if (tag_unique && (tag_unique & m_allowed_unique_mask) == 0)
                continue;

            // force targeted incompatibility: any category of the tag must not conflict with selected ones
            bool ok = true;
            if (m_selected_categories_mask)
            {
                uint64_t tmp = tag_categories;
                while (tmp)
                {
                    uint64_t lsb = tmp & -tmp; // lowest set bit mask
                    uint8_t bit_index = static_cast<uint8_t>(__builtin_ctzll(tmp));
                    if ((reg.incompat_mask_by_bit[bit_index] & m_selected_categories_mask) != 0)
                    {
                        ok = false;
                        break;
                    }
                    tmp ^= lsb;
                }
            }
            if (!ok)
                continue;

            // force exclusive groups: check if tag conflicts with already active group arrays
            if (checkExclusiveGroupsConflict(tag_categories, reg))
                continue;

            // accepted tag: update masks and exclusive groups state
            if (tag_unique)
                m_allowed_unique_mask &= ~tag_unique;
            m_selected_categories_mask |= tag_categories;
            updateExclusiveGroupsState(tag_categories, reg);

            m_tags.push_back(rand_tag);
            m_precomputed_tokens.push_back(rand_tag_info.precomputed_tokens);
            m_tags_set.insert(rand_tag);
        }
    }
};
