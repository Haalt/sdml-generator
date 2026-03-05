#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "models/clip_embed_cache.hpp"
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
    std::vector<int64_t> precomputed_tokens;          // precomputed token IDs
    std::vector<uint32_t> precomputed_clip_tag_ids;   // precomputed CLIP vocab IDs for semantic subtags
    uint64_t categories_mask;                         // structural category membership of the atomic sampled leaf
    uint64_t semantic_categories_mask;                // structural membership plus semantic subtags emitted by this leaf
    uint64_t unique_bits;                             // subset of semantic_categories_mask that are unique
    uint64_t incompat_mask;                           // union of incompatibility masks for all categories in semantic_categories_mask
    bool has_unique_category_conflict = false;        // true when one packed leaf contains multiple subtags from one unique category
    std::vector<uint64_t> exclusive_arrays_masks;     // per exclusive group index: structural side membership
    std::vector<uint64_t> semantic_exclusive_arrays_masks; // per exclusive group index: semantic side membership for split subtags
    std::vector<uint64_t> unique_parent_arrays_masks; // per unique-parent group index: arrays mask of immediate children membership
    std::vector<int32_t> unique_parent_child_index;   // per unique-parent group index: child index (-1 none, -2 mixed)
};

class Tag_list
{
private:
    template <typename Func>
    static void forEachSetBit(uint64_t mask, Func &&fn)
    {
        while (mask)
        {
#if defined(_MSC_VER) && defined(_WIN64)
            unsigned long bit;
            _BitScanForward64(&bit, mask);
            uint8_t bitIndex = static_cast<uint8_t>(bit);
#elif defined(__GNUC__) || defined(__clang__)
            uint8_t bitIndex = static_cast<uint8_t>(__builtin_ctzll(mask));
#else
            uint8_t bitIndex = 0;
            uint64_t temp = mask;
            while ((temp & 1ull) == 0ull)
            {
                ++bitIndex;
                temp >>= 1;
            }
#endif
            fn(bitIndex);
            mask &= (mask - 1);
        }
    }

    // c++17 inline definitions
    inline static std::vector<TagInfo> s_allowed_tags;
    inline static std::once_flag initFlag_;
    inline static std::unordered_map<std::string, size_t> s_index_by_tag;                        // tag -> index in s_allowed_tags
    inline static std::vector<size_t> s_unique_parent_group_array_counts;                        // number of children per unique parent group
    inline static uint64_t s_parent_unique_bits = 0;                                             // bits corresponding to unique map-node categories
    inline static std::array<std::vector<size_t>, 64> s_tags_by_unique_bit;                      // unique category bit -> tag indices
    inline static std::array<std::vector<size_t>, 64> s_tags_by_incompat_bit;                    // incompatibility bit -> tag indices
    inline static std::vector<std::vector<std::vector<size_t>>> s_exclusive_group_array_to_tags; // group -> array -> tag indices
    inline static std::vector<std::vector<std::vector<size_t>>> s_unique_parent_child_to_tags;   // group -> child -> tag indices

    uint64_t m_allowed_unique_mask = 0;      // Allowed unique category bits remaining
    uint64_t m_selected_categories_mask = 0; // All categories already selected (for incompatibility checks)

    // exclusive groups state: allowed arrays mask per group index; starts with all arrays allowed and narrows as tags are chosen
    std::vector<uint64_t> m_allowed_arrays_masks; // index aligned with reg.exclusive_groups
    // unique parent map groups state: same as above but for unique map categories
    std::vector<uint64_t> m_unique_parent_allowed_arrays_masks; // index aligned with s_unique_parent_group_array_counts

    std::vector<std::string> m_tags;
    std::vector<std::vector<int64_t>> m_precomputed_tokens; // parallel to m_tags
    std::vector<std::vector<uint32_t>> m_precomputed_clip_tag_ids; // parallel to m_tags
    std::unordered_set<std::string> m_tags_set;             // for quick lookup

    std::vector<size_t> m_candidate_indices;
    std::vector<size_t> m_candidate_positions;
    std::vector<uint8_t> m_is_candidate_active;
    size_t m_active_candidate_count{0};

    static uint64_t computeIncompatibilityMask(const CategoryRegistry &reg, uint64_t categoriesMask)
    {
        uint64_t incompatibilityMask = 0;
        uint64_t tmp = categoriesMask;
        while (tmp)
        {
            uint64_t lsb = tmp & -tmp;
            uint8_t bitIndex = static_cast<uint8_t>(__builtin_ctzll(tmp));
            incompatibilityMask |= reg.incompat_mask_by_bit[bitIndex];
            tmp ^= lsb;
        }
        return incompatibilityMask;
    }

    static uint64_t computeExclusiveGroupMask(uint64_t categoriesMask, const CategoryRegistry::ExclusiveGroup &group)
    {
        uint64_t mask = 0;
        for (size_t arrayIndex = 0; arrayIndex < group.group_masks.size(); ++arrayIndex)
        {
            if ((categoriesMask & group.group_masks[arrayIndex]) != 0)
            {
                mask |= (uint64_t(1) << arrayIndex);
            }
        }
        return mask;
    }

    static bool hasUniqueCategoryConflict(const std::string &tag,
                                          const std::unordered_map<std::string, uint64_t> &semanticCategoryMaskByTag,
                                          uint64_t uniqueMask)
    {
        std::array<uint8_t, 64> counts{};
        std::unordered_set<std::string> seenSemanticTags;
        for (const auto &semanticTag : generator::Tokenizer::splitTagString(tag))
        {
            if (!seenSemanticTags.insert(semanticTag).second)
                continue;
            auto itMask = semanticCategoryMaskByTag.find(semanticTag);
            if (itMask == semanticCategoryMaskByTag.end())
                continue;
            uint64_t mask = itMask->second & uniqueMask;
            forEachSetBit(mask, [&](uint8_t bit)
                          { ++counts[bit]; });
        }
        for (uint8_t count : counts)
        {
            if (count > 1)
                return true;
        }
        return false;
    }

    static uint64_t getEffectiveExclusiveArraysMask(const TagInfo &info, size_t groupIndex)
    {
        const uint64_t structuralMask = info.exclusive_arrays_masks[groupIndex];
        if (structuralMask != 0)
        {
            return structuralMask;
        }
        return info.semantic_exclusive_arrays_masks[groupIndex];
    }

    // helper method to check if a tag's categories conflict with exclusive groups
    bool checkExclusiveGroupsConflict(const TagInfo &info, const CategoryRegistry &reg) const
    {
        const size_t group_count = reg.exclusive_groups.size();
        for (size_t gi = 0; gi < group_count; ++gi)
        {
            const uint64_t allowed_mask = m_allowed_arrays_masks[gi];
            const uint64_t arrays_with_tag_mask = getEffectiveExclusiveArraysMask(info, gi);
            if (arrays_with_tag_mask == 0)
                continue;
            if ((allowed_mask & arrays_with_tag_mask) == 0)
                return true;
        }
        // enforce unique-parent map groups
        for (size_t gi = 0; gi < m_unique_parent_allowed_arrays_masks.size(); ++gi)
        {
            const uint64_t allowed_mask = m_unique_parent_allowed_arrays_masks[gi];
            const uint64_t arrays_with_tag_mask = info.unique_parent_arrays_masks[gi];
            if (arrays_with_tag_mask == 0)
                continue;
            if ((allowed_mask & arrays_with_tag_mask) == 0)
                return true;
        }
        return false;
    }

    // update exclusive groups state when adding a tag
    void updateExclusiveGroupsState(const TagInfo &info,
                                    const CategoryRegistry &reg,
                                    std::vector<std::pair<size_t, uint64_t>> &removedExclusiveMasks,
                                    std::vector<std::pair<size_t, uint64_t>> &removedUniqueParentMasks)
    {
        const size_t group_count = reg.exclusive_groups.size();
        for (size_t gi = 0; gi < group_count; ++gi)
        {
            const uint64_t before = m_allowed_arrays_masks[gi];
            const uint64_t arrays_with_tag_mask = getEffectiveExclusiveArraysMask(info, gi);
            if (arrays_with_tag_mask != 0)
                m_allowed_arrays_masks[gi] &= arrays_with_tag_mask;
            const uint64_t removed = before & ~m_allowed_arrays_masks[gi];
            if (removed != 0)
                removedExclusiveMasks.emplace_back(gi, removed);
        }
        // narrow unique-parent groups
        for (size_t gi = 0; gi < m_unique_parent_allowed_arrays_masks.size(); ++gi)
        {
            const uint64_t before = m_unique_parent_allowed_arrays_masks[gi];
            const uint64_t arrays_with_tag_mask = info.unique_parent_arrays_masks[gi];
            if (arrays_with_tag_mask != 0)
                m_unique_parent_allowed_arrays_masks[gi] &= arrays_with_tag_mask;
            const uint64_t removed = before & ~m_unique_parent_allowed_arrays_masks[gi];
            if (removed != 0)
                removedUniqueParentMasks.emplace_back(gi, removed);
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
    std::unordered_map<std::string, uint64_t> semanticCategoryMaskByTag;
    for (auto &vec : s_tags_by_unique_bit)
        vec.clear();
    for (auto &vec : s_tags_by_incompat_bit)
        vec.clear();
    for (const auto &kv : reg.bit_by_category)
    {
      const std::string &category_path = kv.first;
      const uint8_t bit_index = kv.second;
      uint64_t bit = (uint64_t(1) << bit_index);

      std::vector<std::string> tags = tg->getListNode(category_path);
      for (const auto &tag : tags)
      {
        tag_info_map[tag].categories_mask |= bit;
        for (const auto &semanticTag : generator::Tokenizer::splitTagString(tag))
        {
          semanticCategoryMaskByTag[semanticTag] |= bit;
        }
      }
    }

    // Build unique-parent groups from rules.unique_categories that point to map nodes
    const Rules &rules = tg->getRules();
    std::vector<std::vector<std::unordered_set<std::string>>> unique_parent_child_sets; // per group: per child: set of tags
    s_unique_parent_group_array_counts.clear();
    s_parent_unique_bits = 0;
    for (const auto &path : rules.unique_categories)
    {
      auto node2 = tg->getNode(path);
      auto mapNode = std::dynamic_pointer_cast<MapNode>(node2);
      if (!mapNode)
        continue;

      std::vector<std::unordered_set<std::string>> childSets;
      childSets.reserve(mapNode->children.size());
      for (const auto &child : mapNode->children)
      {
        const std::string childPath = path + "." + child.first;
        std::vector<std::string> tags = tg->getListNode(childPath);
        childSets.emplace_back(tags.begin(), tags.end());
      }
      if (!childSets.empty())
      {
        s_unique_parent_group_array_counts.push_back(childSets.size());
        unique_parent_child_sets.push_back(std::move(childSets));
      }

      // record parent unique bit so we don't consume it like a leaf-unique bit
      const CategoryRegistry &reg2 = tg->getRegistry();
      auto itb = reg2.bit_by_category.find(path);
      if (itb != reg2.bit_by_category.end())
      {
        s_parent_unique_bits |= (uint64_t(1) << itb->second);
      }
    }

    // Precompute exclusivity membership masks per group for each tag
    const size_t group_count = reg.exclusive_groups.size();
    const size_t unique_parent_group_count = unique_parent_child_sets.size();

    s_exclusive_group_array_to_tags.clear();
    s_exclusive_group_array_to_tags.resize(group_count);
    for (size_t gi = 0; gi < group_count; ++gi)
    {
      const size_t arrayCount = reg.exclusive_groups[gi].group_masks.size();
      s_exclusive_group_array_to_tags[gi].resize(arrayCount);
    }
    s_unique_parent_child_to_tags.clear();
    s_unique_parent_child_to_tags.resize(unique_parent_group_count);
    for (size_t gi = 0; gi < unique_parent_group_count; ++gi)
    {
      s_unique_parent_child_to_tags[gi].resize(unique_parent_child_sets[gi].size());
    }
    // build s_allowed_tags from tag_info_map
    s_allowed_tags.reserve(tag_info_map.size());
    s_index_by_tag.reserve(tag_info_map.size());
    for (const auto &pair : tag_info_map)
    {
      TagInfo info = pair.second;
      info.semantic_categories_mask = info.categories_mask;
      for (const auto &semanticTag : generator::Tokenizer::splitTagString(info.tag))
      {
        auto itMask = semanticCategoryMaskByTag.find(semanticTag);
        if (itMask != semanticCategoryMaskByTag.end())
        {
          info.semantic_categories_mask |= itMask->second;
        }
      }
      // Do not consume map-node unique bits; those are enforced by unique-parent narrowing
      const uint64_t leafUniqueMask = (reg.unique_mask_all & ~s_parent_unique_bits);
      info.unique_bits = (info.semantic_categories_mask & leafUniqueMask);
      info.has_unique_category_conflict = hasUniqueCategoryConflict(info.tag, semanticCategoryMaskByTag, leafUniqueMask);
      info.incompat_mask = computeIncompatibilityMask(reg, info.semantic_categories_mask);

      // per-group arrays mask participation
      info.exclusive_arrays_masks.assign(group_count, 0);
      info.semantic_exclusive_arrays_masks.assign(group_count, 0);
      for (size_t gi = 0; gi < group_count; ++gi)
      {
        const auto &eg = reg.exclusive_groups[gi];
        info.exclusive_arrays_masks[gi] = computeExclusiveGroupMask(info.categories_mask, eg);
        info.semantic_exclusive_arrays_masks[gi] = computeExclusiveGroupMask(info.semantic_categories_mask, eg);
      }

      // unique-parent groups membership
      info.unique_parent_arrays_masks.assign(unique_parent_group_count, 0);
      info.unique_parent_child_index.assign(unique_parent_group_count, -1);
      for (size_t gi = 0; gi < unique_parent_group_count; ++gi)
      {
        uint64_t mask = 0;
        const auto &childSets = unique_parent_child_sets[gi];
        for (size_t ai = 0; ai < childSets.size(); ++ai)
        {
          if (childSets[ai].count(info.tag))
            mask |= (uint64_t(1) << ai);
        }
        info.unique_parent_arrays_masks[gi] = mask;
        if (mask != 0)
        {
          if ((mask & (mask - 1)) == 0)
          {
            info.unique_parent_child_index[gi] = static_cast<int32_t>(__builtin_ctzll(mask));
          }
          else
          {
            info.unique_parent_child_index[gi] = -2; // belongs to multiple children
          }
        }
      }

      size_t tagIndex = s_allowed_tags.size();
      s_index_by_tag.emplace(pair.first, tagIndex);
      s_allowed_tags.push_back(std::move(info));

      const TagInfo &stored = s_allowed_tags.back();
      forEachSetBit(stored.unique_bits, [&](uint8_t bit)
                    { s_tags_by_unique_bit[bit].push_back(tagIndex); });

      forEachSetBit(stored.incompat_mask, [&](uint8_t bit)
                    { s_tags_by_incompat_bit[bit].push_back(tagIndex); });

      for (size_t gi = 0; gi < group_count; ++gi)
      {
        uint64_t mask = getEffectiveExclusiveArraysMask(stored, gi);
        forEachSetBit(mask, [&](uint8_t arrayIdx)
                      {
                        if (arrayIdx < s_exclusive_group_array_to_tags[gi].size())
                          s_exclusive_group_array_to_tags[gi][arrayIdx].push_back(tagIndex);
                      });
      }

      for (size_t gi = 0; gi < unique_parent_group_count; ++gi)
      {
        int32_t childIndex = s_allowed_tags.back().unique_parent_child_index[gi];
        if (childIndex >= 0)
        {
          s_unique_parent_child_to_tags[gi][static_cast<size_t>(childIndex)].push_back(tagIndex);
        }
      }
    } });
    }

    bool canSelectTag(const TagInfo &info, const CategoryRegistry &reg) const
    {
        if (m_tags_set.count(info.tag))
            return false;

        if (info.has_unique_category_conflict)
            return false;

        if (info.unique_bits && (info.unique_bits & m_allowed_unique_mask) == 0)
            return false;

        if (m_selected_categories_mask && (info.incompat_mask & m_selected_categories_mask))
            return false;

        if (checkExclusiveGroupsConflict(info, reg))
            return false;

        return true;
    }

    void deactivate_candidate_at_position(size_t pos)
    {
        if (pos >= m_active_candidate_count)
            return;

        size_t tagIdx = m_candidate_indices[pos];
        size_t lastPos = m_active_candidate_count - 1;

        if (pos != lastPos)
        {
            size_t lastTagIdx = m_candidate_indices[lastPos];
            m_candidate_indices[pos] = lastTagIdx;
            m_candidate_positions[lastTagIdx] = pos;
        }

        m_active_candidate_count = lastPos;
        m_is_candidate_active[tagIdx] = 0;
        m_candidate_positions[tagIdx] = std::numeric_limits<size_t>::max();
    }

    void deactivate_candidate(size_t tagIdx)
    {
        if (tagIdx >= m_candidate_positions.size())
            return;

        if (!m_is_candidate_active[tagIdx])
            return;

        size_t pos = m_candidate_positions[tagIdx];
        if (pos >= m_active_candidate_count)
            return;

        deactivate_candidate_at_position(pos);
    }

    void prune_candidates(const CategoryRegistry &reg)
    {
        const auto &allowedTagsRef = getAllowedTags();
        size_t i = 0;
        while (i < m_active_candidate_count)
        {
            size_t tagIdx = m_candidate_indices[i];
            const TagInfo &info = allowedTagsRef[tagIdx];
            if (!canSelectTag(info, reg))
            {
                deactivate_candidate_at_position(i);
                continue;
            }
            ++i;
        }
    }

    void add_tag_by_index(size_t idx, const CategoryRegistry &reg)
    {
        const auto &allowedTags = getAllowedTags();
        if (idx >= allowedTags.size())
            return;

        const TagInfo &info = allowedTags[idx];

        if (m_tags_set.insert(info.tag).second)
        {
            m_tags.push_back(info.tag);
            m_precomputed_tokens.push_back(info.precomputed_tokens);
            m_precomputed_clip_tag_ids.push_back(info.precomputed_clip_tag_ids);
        }

        uint64_t previousUniqueMask = m_allowed_unique_mask;
        uint64_t previousSelectedMask = m_selected_categories_mask;

        if (info.unique_bits)
            m_allowed_unique_mask &= ~info.unique_bits;

        uint64_t newlyActivatedCategoryBits = info.semantic_categories_mask & ~previousSelectedMask;

        std::vector<std::pair<size_t, uint64_t>> removedExclusiveMasks;
        std::vector<std::pair<size_t, uint64_t>> removedUniqueParentMasks;
        removedExclusiveMasks.reserve(4);
        removedUniqueParentMasks.reserve(4);

        updateExclusiveGroupsState(info, reg, removedExclusiveMasks, removedUniqueParentMasks);

        m_selected_categories_mask |= info.semantic_categories_mask;

        deactivate_candidate(idx);

        uint64_t consumedUniqueBits = previousUniqueMask & ~m_allowed_unique_mask;
        const auto &tagsByUniqueBit = getTagsByUniqueBit();
        const auto &tagsByIncompatBit = getTagsByIncompatBit();
        const auto &exclusiveGroupArrayToTags = getExclusiveGroupArrayToTags();
        const auto &uniqueParentChildToTags = getUniqueParentChildToTags();
        const auto &allowedTagsRef = getAllowedTags();

        forEachSetBit(consumedUniqueBits, [&](uint8_t bit)
                      {
                          for (size_t tagIndex : tagsByUniqueBit[bit])
                              deactivate_candidate(tagIndex); });

        forEachSetBit(newlyActivatedCategoryBits, [&](uint8_t bit)
                      {
                          for (size_t tagIndex : tagsByIncompatBit[bit])
                              deactivate_candidate(tagIndex); });

        for (const auto &entry : removedExclusiveMasks)
        {
            const size_t groupIndex = entry.first;
            const uint64_t removedMask = entry.second;
            forEachSetBit(removedMask, [&](uint8_t arrayIdx)
                          {
                              if (groupIndex >= exclusiveGroupArrayToTags.size())
                                  return;
                              if (arrayIdx >= exclusiveGroupArrayToTags[groupIndex].size())
                                  return;
                              const auto &tagIndices = exclusiveGroupArrayToTags[groupIndex][arrayIdx];
                              for (size_t candidateIdx : tagIndices)
                              {
                                  if (candidateIdx >= m_is_candidate_active.size() || !m_is_candidate_active[candidateIdx])
                                      continue;
                                  const TagInfo &candidateInfo = allowedTagsRef[candidateIdx];
                                  if ((getEffectiveExclusiveArraysMask(candidateInfo, groupIndex) & m_allowed_arrays_masks[groupIndex]) == 0)
                                      deactivate_candidate(candidateIdx);
                              } });
        }

        for (const auto &entry : removedUniqueParentMasks)
        {
            const size_t groupIndex = entry.first;
            const uint64_t removedMask = entry.second;
            forEachSetBit(removedMask, [&](uint8_t arrayIdx)
                          {
                              if (groupIndex >= uniqueParentChildToTags.size())
                                  return;
                              if (arrayIdx >= uniqueParentChildToTags[groupIndex].size())
                                  return;
                              const auto &tagIndices = uniqueParentChildToTags[groupIndex][arrayIdx];
                              for (size_t candidateIdx : tagIndices)
                              {
                                  if (candidateIdx >= m_is_candidate_active.size() || !m_is_candidate_active[candidateIdx])
                                      continue;
                                  const TagInfo &candidateInfo = allowedTagsRef[candidateIdx];
                                  if ((candidateInfo.unique_parent_arrays_masks[groupIndex] & m_unique_parent_allowed_arrays_masks[groupIndex]) == 0)
                                      deactivate_candidate(candidateIdx);
                              } });
        }
    }

    bool pick_tag_from_child(size_t groupIndex, int32_t childIndex, const CategoryRegistry &reg, size_t &outTagIdx)
    {
        const auto &uniqueParentChildToTags = getUniqueParentChildToTags();
        if (groupIndex >= uniqueParentChildToTags.size())
            return false;

        if (childIndex < 0)
            return false;

        const auto &childList = uniqueParentChildToTags[groupIndex][static_cast<size_t>(childIndex)];
        if (childList.empty())
            return false;

        size_t childSize = childList.size();
        if (childSize == 0)
            return false;

        const auto &allowedTagsRef = getAllowedTags();
        int maxIndex = static_cast<int>(childSize - 1);
        int randomStart = (maxIndex >= 0) ? rng::irange(0, maxIndex) : 0;
        for (size_t offset = 0; offset < childSize; ++offset)
        {
            size_t candidateIdx = childList[(static_cast<size_t>(randomStart) + offset) % childSize];
            if (candidateIdx >= m_is_candidate_active.size())
                continue;
            if (!m_is_candidate_active[candidateIdx])
                continue;
            const TagInfo &info = allowedTagsRef[candidateIdx];
            if (!canSelectTag(info, reg))
                continue;
            outTagIdx = candidateIdx;
            return true;
        }
        return false;
    }

    void attempt_child_followups(size_t tagIdx, const CategoryRegistry &reg, size_t targetCount)
    {
        static constexpr size_t MAX_CHILD_FOLLOWUPS = 3;
        const TagInfo &info = getAllowedTags()[tagIdx];

        for (size_t gi = 0; gi < info.unique_parent_child_index.size(); ++gi)
        {
            int32_t childIndex = info.unique_parent_child_index[gi];
            if (childIndex < 0)
                continue;

            size_t remainingTarget = (targetCount > m_tags.size()) ? (targetCount - m_tags.size()) : 0;
            if (remainingTarget == 0)
                return;

            size_t followupLimit = std::min(MAX_CHILD_FOLLOWUPS, remainingTarget);

            for (size_t count = 0; count < followupLimit && m_tags.size() < targetCount; ++count)
            {
                size_t childTagIdx = 0;
                if (!pick_tag_from_child(gi, childIndex, reg, childTagIdx))
                    break;

                add_tag_by_index(childTagIdx, reg);
            }
        }
    }

public:
    // Instance-level storage for when a specific Tag_dict is provided
    Tag_dict *m_tag_dict = nullptr;  // Non-owning pointer to instance Tag_dict (null = use static)
    std::vector<TagInfo> m_instance_allowed_tags;
    std::unordered_map<std::string, size_t> m_instance_index_by_tag;
    std::vector<size_t> m_instance_unique_parent_group_array_counts;
    uint64_t m_instance_parent_unique_bits = 0;
    std::array<std::vector<size_t>, 64> m_instance_tags_by_unique_bit;
    std::array<std::vector<size_t>, 64> m_instance_tags_by_incompat_bit;
    std::vector<std::vector<std::vector<size_t>>> m_instance_exclusive_group_array_to_tags;
    std::vector<std::vector<std::vector<size_t>>> m_instance_unique_parent_child_to_tags;

    // Accessors for static vs instance data
    const std::vector<TagInfo> &getAllowedTags() const
    {
        return m_tag_dict ? m_instance_allowed_tags : s_allowed_tags;
    }
    const std::unordered_map<std::string, size_t> &getIndexByTag() const
    {
        return m_tag_dict ? m_instance_index_by_tag : s_index_by_tag;
    }
    const std::vector<size_t> &getUniqueParentGroupArrayCounts() const
    {
        return m_tag_dict ? m_instance_unique_parent_group_array_counts : s_unique_parent_group_array_counts;
    }
    uint64_t getParentUniqueBits() const
    {
        return m_tag_dict ? m_instance_parent_unique_bits : s_parent_unique_bits;
    }
    const std::array<std::vector<size_t>, 64> &getTagsByUniqueBit() const
    {
        return m_tag_dict ? m_instance_tags_by_unique_bit : s_tags_by_unique_bit;
    }
    const std::array<std::vector<size_t>, 64> &getTagsByIncompatBit() const
    {
        return m_tag_dict ? m_instance_tags_by_incompat_bit : s_tags_by_incompat_bit;
    }
    const std::vector<std::vector<std::vector<size_t>>> &getExclusiveGroupArrayToTags() const
    {
        return m_tag_dict ? m_instance_exclusive_group_array_to_tags : s_exclusive_group_array_to_tags;
    }
    const std::vector<std::vector<std::vector<size_t>>> &getUniqueParentChildToTags() const
    {
        return m_tag_dict ? m_instance_unique_parent_child_to_tags : s_unique_parent_child_to_tags;
    }

    void initialize_instance_tags(const generator::Tokenizer *tokenizer,
                                  const generator::ClipEmbedCache *clipEmbedCache = nullptr)
    {
        if (!m_tag_dict)
            return;

        m_instance_allowed_tags.clear();
        m_instance_index_by_tag.clear();

        std::unordered_map<std::string, TagInfo> tag_info_map;
        std::vector<std::string> all_tags = m_tag_dict->flatten();

        for (const auto &tag : all_tags)
        {
            TagInfo info;
            info.tag = tag;
            info.categories_mask = 0;
            if (tokenizer)
            {
                info.precomputed_tokens = tokenizer->precomputeTagTokens(tag);
            }
            if (clipEmbedCache)
            {
                const std::vector<std::string> semanticTags = generator::Tokenizer::splitTagString(tag);
                info.precomputed_clip_tag_ids.reserve(semanticTags.size());
                for (const std::string &semanticTag : semanticTags)
                {
                    const std::optional<uint32_t> clipTagId = clipEmbedCache->lookupTagId(semanticTag);
                    if (clipTagId.has_value())
                    {
                        info.precomputed_clip_tag_ids.push_back(*clipTagId);
                    }
                }
            }
            tag_info_map[tag] = std::move(info);
        }

        const CategoryRegistry &reg = m_tag_dict->getRegistry();
        std::unordered_map<std::string, uint64_t> semanticCategoryMaskByTag;
        for (auto &vec : m_instance_tags_by_unique_bit)
            vec.clear();
        for (auto &vec : m_instance_tags_by_incompat_bit)
            vec.clear();

        for (const auto &kv : reg.bit_by_category)
        {
            const std::string &category_path = kv.first;
            const uint8_t bit_index = kv.second;
            uint64_t bit = (uint64_t(1) << bit_index);
            std::vector<std::string> tags = m_tag_dict->getListNode(category_path);
            for (const auto &tag : tags)
            {
                tag_info_map[tag].categories_mask |= bit;
                for (const auto &semanticTag : generator::Tokenizer::splitTagString(tag))
                {
                    semanticCategoryMaskByTag[semanticTag] |= bit;
                }
            }
        }

        const Rules &rules = m_tag_dict->getRules();
        std::vector<std::vector<std::unordered_set<std::string>>> unique_parent_child_sets;
        m_instance_unique_parent_group_array_counts.clear();
        m_instance_parent_unique_bits = 0;

        for (const auto &path : rules.unique_categories)
        {
            auto node2 = m_tag_dict->getNode(path);
            auto mapNode = std::dynamic_pointer_cast<MapNode>(node2);
            if (!mapNode)
                continue;
            std::vector<std::unordered_set<std::string>> childSets;
            childSets.reserve(mapNode->children.size());
            for (const auto &child : mapNode->children)
            {
                const std::string childPath = path + "." + child.first;
                std::vector<std::string> tags = m_tag_dict->getListNode(childPath);
                childSets.emplace_back(tags.begin(), tags.end());
            }
            if (!childSets.empty())
            {
                m_instance_unique_parent_group_array_counts.push_back(childSets.size());
                unique_parent_child_sets.push_back(std::move(childSets));
            }
            auto itb = reg.bit_by_category.find(path);
            if (itb != reg.bit_by_category.end())
            {
                m_instance_parent_unique_bits |= (uint64_t(1) << itb->second);
            }
        }

        const size_t group_count = reg.exclusive_groups.size();
        const size_t unique_parent_group_count = unique_parent_child_sets.size();

        m_instance_exclusive_group_array_to_tags.clear();
        m_instance_exclusive_group_array_to_tags.resize(group_count);
        for (size_t gi = 0; gi < group_count; ++gi)
        {
            const size_t arrayCount = reg.exclusive_groups[gi].group_masks.size();
            m_instance_exclusive_group_array_to_tags[gi].resize(arrayCount);
        }
        m_instance_unique_parent_child_to_tags.clear();
        m_instance_unique_parent_child_to_tags.resize(unique_parent_group_count);
        for (size_t gi = 0; gi < unique_parent_group_count; ++gi)
        {
            m_instance_unique_parent_child_to_tags[gi].resize(unique_parent_child_sets[gi].size());
        }

        m_instance_allowed_tags.reserve(tag_info_map.size());
        m_instance_index_by_tag.reserve(tag_info_map.size());

        for (const auto &pair : tag_info_map)
        {
            TagInfo info = pair.second;
            info.semantic_categories_mask = info.categories_mask;
            for (const auto &semanticTag : generator::Tokenizer::splitTagString(info.tag))
            {
                auto itMask = semanticCategoryMaskByTag.find(semanticTag);
                if (itMask != semanticCategoryMaskByTag.end())
                {
                    info.semantic_categories_mask |= itMask->second;
                }
            }
            const uint64_t leafUniqueMask = (reg.unique_mask_all & ~m_instance_parent_unique_bits);
            info.unique_bits = (info.semantic_categories_mask & leafUniqueMask);
            info.has_unique_category_conflict = hasUniqueCategoryConflict(info.tag, semanticCategoryMaskByTag, leafUniqueMask);
            info.incompat_mask = computeIncompatibilityMask(reg, info.semantic_categories_mask);

            info.exclusive_arrays_masks.assign(group_count, 0);
            info.semantic_exclusive_arrays_masks.assign(group_count, 0);
            for (size_t gi = 0; gi < group_count; ++gi)
            {
                const auto &eg = reg.exclusive_groups[gi];
                info.exclusive_arrays_masks[gi] = computeExclusiveGroupMask(info.categories_mask, eg);
                info.semantic_exclusive_arrays_masks[gi] = computeExclusiveGroupMask(info.semantic_categories_mask, eg);
            }

            info.unique_parent_arrays_masks.assign(unique_parent_group_count, 0);
            info.unique_parent_child_index.assign(unique_parent_group_count, -1);
            for (size_t gi = 0; gi < unique_parent_group_count; ++gi)
            {
                uint64_t mask = 0;
                const auto &childSets = unique_parent_child_sets[gi];
                for (size_t ai = 0; ai < childSets.size(); ++ai)
                {
                    if (childSets[ai].count(info.tag))
                        mask |= (uint64_t(1) << ai);
                }
                info.unique_parent_arrays_masks[gi] = mask;
                if (mask != 0)
                {
                    if ((mask & (mask - 1)) == 0)
                    {
                        info.unique_parent_child_index[gi] = static_cast<int32_t>(__builtin_ctzll(mask));
                    }
                    else
                    {
                        info.unique_parent_child_index[gi] = -2;
                    }
                }
            }

            size_t tagIndex = m_instance_allowed_tags.size();
            m_instance_index_by_tag.emplace(pair.first, tagIndex);
            m_instance_allowed_tags.push_back(std::move(info));

            const TagInfo &stored = m_instance_allowed_tags.back();
            forEachSetBit(stored.unique_bits, [&](uint8_t bit)
                          { m_instance_tags_by_unique_bit[bit].push_back(tagIndex); });

            forEachSetBit(stored.incompat_mask, [&](uint8_t bit)
                          { m_instance_tags_by_incompat_bit[bit].push_back(tagIndex); });

            for (size_t gi = 0; gi < group_count; ++gi)
            {
                uint64_t mask = getEffectiveExclusiveArraysMask(stored, gi);
                forEachSetBit(mask, [&](uint8_t arrayIdx)
                              {
                                  if (arrayIdx < m_instance_exclusive_group_array_to_tags[gi].size())
                                      m_instance_exclusive_group_array_to_tags[gi][arrayIdx].push_back(tagIndex);
                              });
            }

            for (size_t gi = 0; gi < unique_parent_group_count; ++gi)
            {
                int32_t childIndex = m_instance_allowed_tags.back().unique_parent_child_index[gi];
                if (childIndex >= 0)
                {
                    m_instance_unique_parent_child_to_tags[gi][static_cast<size_t>(childIndex)].push_back(tagIndex);
                }
            }
        }
    }

public:
    void reset()
    {
        const CategoryRegistry &reg = m_tag_dict ? m_tag_dict->getRegistry() : Tag_dict::getInstance()->getRegistry();

        m_allowed_unique_mask = reg.unique_mask_all;
        m_selected_categories_mask = 0;

        m_allowed_arrays_masks.clear();
        m_allowed_arrays_masks.reserve(reg.exclusive_groups.size());
        for (const auto &eg : reg.exclusive_groups)
        {
            const size_t n = eg.group_masks.size();
            const uint64_t full_mask = (n >= 64) ? ~uint64_t(0) : ((uint64_t(1) << n) - 1);
            m_allowed_arrays_masks.push_back(full_mask);
        }

        const auto &parentCounts = getUniqueParentGroupArrayCounts();
        m_unique_parent_allowed_arrays_masks.clear();
        m_unique_parent_allowed_arrays_masks.reserve(parentCounts.size());
        for (size_t n : parentCounts)
        {
            const uint64_t full_mask = (n >= 64) ? ~uint64_t(0) : ((uint64_t(1) << n) - 1);
            m_unique_parent_allowed_arrays_masks.push_back(full_mask);
        }

        m_tags.clear();
        m_precomputed_tokens.clear();
        m_precomputed_clip_tag_ids.clear();
        m_tags_set.clear();

        const size_t totalTags = getAllowedTags().size();
        m_candidate_indices.resize(totalTags);
        m_candidate_positions.resize(totalTags);
        m_is_candidate_active.resize(totalTags);

        for (size_t i = 0; i < totalTags; ++i)
        {
            m_candidate_indices[i] = i;
            m_candidate_positions[i] = i;
            m_is_candidate_active[i] = 1;
        }

        m_active_candidate_count = totalTags;
    }

    /**
     * @brief Constructs Tag_list using the global Tag_dict singleton
     */
    Tag_list(const generator::Tokenizer *tokenizer = nullptr)
    {
        m_tags.reserve(100);
        m_precomputed_tokens.reserve(100);
        m_precomputed_clip_tag_ids.reserve(100);

        initialize_allowed_tags(tokenizer);

        reset();
    }

    /**
     * @brief Constructs Tag_list using a specific Tag_dict instance
     * @param tokenizer Tokenizer for precomputing tokens
     * @param tagDict Tag dictionary to use (must outlive this Tag_list)
     */
    Tag_list(const generator::Tokenizer *tokenizer,
             Tag_dict *tagDict,
             const generator::ClipEmbedCache *clipEmbedCache = nullptr)
        : m_tag_dict(tagDict)
    {
        m_tags.reserve(100);
        m_precomputed_tokens.reserve(100);
        m_precomputed_clip_tag_ids.reserve(100);

        if (m_tag_dict)
        {
            initialize_instance_tags(tokenizer, clipEmbedCache);
        }
        else
        {
            initialize_allowed_tags(tokenizer);
        }

        reset();
    }

    const std::vector<std::string> &get_tags(void) const { return m_tags; }
    const std::vector<std::vector<int64_t>> &get_precomputed_tokens(void) const { return m_precomputed_tokens; }
    const std::vector<std::vector<uint32_t>> &get_precomputed_clip_tag_ids(void) const { return m_precomputed_clip_tag_ids; }

    void set_tags(const std::vector<std::string> &tags)
    {
        reset(); // start by reseting the state

        const CategoryRegistry &reg = m_tag_dict ? m_tag_dict->getRegistry() : Tag_dict::getInstance()->getRegistry();
        const auto &indexByTag = getIndexByTag();

        m_tags_set.reserve(tags.size() * 2);

        for (const auto &tag : tags)
        {
            auto it = indexByTag.find(tag);
            if (it != indexByTag.end())
            {
                add_tag_by_index(it->second, reg);
            }
        }
    }

    void insert_tags(const std::vector<std::string> &tags)
    {
        m_tags.insert(m_tags.end(), tags.begin(), tags.end());

        const auto &indexByTag = getIndexByTag();
        const auto &allowedTagsRef = getAllowedTags();

        // add precomputed tokens for the new tags
        for (const auto &tag : tags)
        {
            auto it = indexByTag.find(tag);
            if (it != indexByTag.end())
            {
                const TagInfo &info = allowedTagsRef[it->second];
                m_precomputed_tokens.push_back(info.precomputed_tokens);
                m_precomputed_clip_tag_ids.push_back(info.precomputed_clip_tag_ids);
            }
        }
    }

    void generate_tags(const size_t num = 10)
    {
        const CategoryRegistry &reg = m_tag_dict ? m_tag_dict->getRegistry() : Tag_dict::getInstance()->getRegistry();
        const auto &allowedTagsRef = getAllowedTags();

        while (m_tags.size() < num && m_active_candidate_count > 0)
        {
            size_t drawPos = static_cast<size_t>(rng::irange(0, static_cast<int>(m_active_candidate_count) - 1));
            size_t tagIdx = m_candidate_indices[drawPos];
            const TagInfo &info = allowedTagsRef[tagIdx];

            if (!canSelectTag(info, reg))
            {
                deactivate_candidate_at_position(drawPos);
                continue;
            }

            add_tag_by_index(tagIdx, reg);

            if (m_tags.size() >= num)
                break;

            attempt_child_followups(tagIdx, reg, num);
        }
    }
};
