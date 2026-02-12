#pragma once

#include <algorithm>
#include <iterator>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "models/Lora.h"

// FD
namespace generator
{
    class Tokenizer;
}

class Lora_controller
{
    std::vector<MainLora *> mains_;
    std::vector<SecondaryLora *> secondaries_;
    std::vector<ConditionalVariantLora *> conditionals_;

    std::unordered_map<std::string, TagLora *> tag_lora_map_;
    std::unordered_map<Lora *, std::vector<SecondaryLora *>> main_compatibility_map_;
    std::unordered_map<SecondaryLora *, std::vector<Lora *>> secondary_compatibility_map_;
    std::unordered_map<std::string, std::vector<int64_t>> tag_precompute_cache_;
    std::unordered_map<std::string, std::vector<int64_t>> variation_precompute_cache_;

    size_t size_;

    void invert_map(const std::unordered_map<Lora *, std::vector<SecondaryLora *>> &map)
    {
        std::unordered_map<SecondaryLora *, std::vector<Lora *>> inverted_map;
        for (const auto &el : map)
        {
            for (SecondaryLora *m : el.second)
            {
                inverted_map[m].push_back(el.first); // TODO: check if this bullshit works
            }
        }
        secondary_compatibility_map_ = inverted_map;
    }

public:
    ~Lora_controller(void)
    {
        for (const auto *a : mains_)
            delete a;
        for (const auto *a : secondaries_)
            delete a;
        for (const auto *a : conditionals_)
            delete a;
        std::set<TagLora *> tag_set;
        for (const auto &pair : tag_lora_map_)
            tag_set.insert(pair.second);
        for (TagLora *ptr : tag_set)
            delete ptr;
    }

    // Default constructor for models without LoRAs
    Lora_controller() : size_(0) {}

    Lora_controller(
        std::vector<MainLora *> mains, std::vector<SecondaryLora *> secondaries,
        std::vector<ConditionalVariantLora *> conditionals,
        std::unordered_map<std::string, TagLora *> tag_lora_map,
        std::unordered_map<Lora *, std::vector<SecondaryLora *>> main_compatibility_map)
        : mains_(mains), secondaries_(secondaries), conditionals_(conditionals),
          tag_lora_map_(tag_lora_map),
          main_compatibility_map_(main_compatibility_map)
    {
        invert_map(main_compatibility_map_);

        // pre compute the selection_arr's size
        size_ = mains_.size() + secondaries_.size() + conditionals_.size() + tag_lora_map_.size();
    }

    Lora_controller(const Lora_controller &) = delete;
    Lora_controller &operator=(const Lora_controller &) = delete;

    // just reroll every LoRA
    void reroll_loras(void)
    {
        for (auto *a : mains_)
            a->reroll();
        for (auto *a : secondaries_)
            a->reroll();
        for (auto *a : conditionals_)
            a->reroll();
        for (auto &a : tag_lora_map_)
            a.second->reroll(); // multiple rerolls per TagLora
    }

    std::unordered_map<int64_t, Lora *> id_to_lora_map_;

    // initialize precomputed tokens for all LoRAs
    void precompute_all_tokens(const generator::Tokenizer *tokenizer)
    {
        if (!tokenizer)
            return;

        id_to_lora_map_.clear();

        auto process = [&](Lora *lora)
        {
            lora->precompute_tokens(
                tokenizer,
                &tag_precompute_cache_,
                &variation_precompute_cache_);
            std::optional<std::pair<int64_t, float>> loraToken = lora->get_precomputed_lora_token();
            if (loraToken.has_value())
            {
                id_to_lora_map_[loraToken->first] = lora;
            }
        };

        for (auto *lora : mains_)
            process(lora);
        for (auto *lora : secondaries_)
            process(lora);
        for (auto *lora : conditionals_)
            process(lora);
        for (auto &pair : tag_lora_map_)
            process(pair.second);
    }

    size_t get_tag_precompute_cache_size(void) const
    {
        return tag_precompute_cache_.size();
    }

    size_t get_variation_precompute_cache_size(void) const
    {
        return variation_precompute_cache_.size();
    }

    std::vector<int64_t> sanitize_loras(const std::vector<int64_t> &input_ids)
    {
        std::vector<int64_t> result;
        Lora *selected_main = nullptr;

        // 1. Identify Main LoRA
        for (int64_t id : input_ids)
        {
            if (id_to_lora_map_.count(id))
            {
                Lora *l = id_to_lora_map_[id];
                if (l->get_type() == Lora::MAIN)
                {
                    if (!selected_main)
                    {
                        selected_main = l;
                        result.push_back(id);
                    }
                }
            }
        }

        // 2. Filter Secondaries/Conditionals based on Main
        for (int64_t id : input_ids)
        {
            if (id_to_lora_map_.count(id))
            {
                Lora *l = id_to_lora_map_[id];
                if (l->get_type() == Lora::MAIN)
                    continue; // Already handled

                bool allowed = true;
                if (selected_main && l->get_type() == Lora::SECONDARY)
                {
                    const auto &compatible = main_compatibility_map_[selected_main];
                    bool found = false;
                    for (auto *sec : compatible)
                    {
                        if (sec == l)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                        allowed = false;
                }

                if (allowed)
                {
                    result.push_back(id);
                }
            }
            else
            {
                // Keep unknown IDs (e.g. synthetic tag IDs)
                result.push_back(id);
            }
        }
        return result;
    }

    const Lora *get_lora_by_id(int64_t id) const
    {
        auto it = id_to_lora_map_.find(id);
        if (it != id_to_lora_map_.end())
            return it->second;
        return nullptr;
    }

    const Lora *get_lora_by_name(const std::string &name) const
    {
        for (auto *a : mains_)
            if (a->get_lora_tag() == name)
                return a;
        for (auto *a : secondaries_)
            if (a->get_lora_tag() == name)
                return a;
        for (auto *a : conditionals_)
            if (a->get_lora_tag() == name)
                return a;
        for (auto &[_, l] : tag_lora_map_)
            if (l->get_lora_tag() == name)
                return l;
        return nullptr;
    }

    std::vector<Lora *> get_loras(Lora *selected_main = nullptr,
                                  std::unordered_set<Lora *> loras = {},
                                  Lora *selected_conditional = nullptr)
    {

        std::vector<Lora *> selection_arr;
        selection_arr.reserve(size_);

        if (selected_conditional == nullptr)
            selection_arr.insert(selection_arr.end(), conditionals_.begin(), conditionals_.end());

        if (selected_main != nullptr)
        {
            selection_arr.insert(selection_arr.end(),
                                 main_compatibility_map_[selected_main].begin(),
                                 main_compatibility_map_[selected_main].end());
            return selection_arr;
        }

        if (loras.size() == 0)
        {
            selection_arr.insert(selection_arr.end(), secondaries_.begin(), secondaries_.end());
            selection_arr.insert(selection_arr.end(), mains_.begin(), mains_.end());
            return selection_arr;
        }
        else
        {
            std::vector<std::vector<Lora *>> secondaries;
            for (const auto l : loras)
            {
                if (l->get_type() == Lora::SECONDARY)
                {
                    SecondaryLora *m = dynamic_cast<SecondaryLora *>(l);
                    if (m == nullptr)
                        continue;

                    std::vector<Lora *> lora_vec;
                    for (Lora *action_ptr : secondary_compatibility_map_[m])
                        lora_vec.push_back(action_ptr);

                    secondaries.push_back(lora_vec);
                }
            }
            std::vector<Lora *> common_elements = get_common_elements(secondaries);
            selection_arr.insert(selection_arr.end(), common_elements.begin(), common_elements.end());
        }

        return selection_arr;
    }

    std::vector<Lora *> get_common_elements(const std::vector<std::vector<Lora *>> mat)
    {
        if (mat.size() < 2)
            return mat[0];
        std::set<Lora *> common_elements(mat[0].begin(), mat[0].end());

        for (size_t i = 1; i < mat.size(); ++i)
        {
            std::set<Lora *> tmp;
            std::set_intersection(common_elements.begin(), common_elements.end(),
                                  mat[i].begin(), mat[i].end(),
                                  std::inserter(tmp, tmp.begin()));
            common_elements.swap(tmp);
            if (common_elements.empty())
                break;
        }

        return std::vector<Lora *>(common_elements.begin(), common_elements.end());
    }

    const std::unordered_map<std::string, TagLora *> &get_tag_lora_map(void) const
    {
        return tag_lora_map_;
    }
};

Lora_controller *init_lora_controller_from_json(const std::string &json_path);
