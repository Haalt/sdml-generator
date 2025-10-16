#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "models/Lora.h"
#include "models/Lora_controller.h"
#include "utils/fast_rand.hpp"

using nlohmann::json;

namespace
{
    struct LoraFields
    {
        std::string lora_tag;
        std::string prefix_tag = "lora";
        float min_weight = 1.f;
        float max_weight = 1.f;
        std::vector<std::string> const_tags;
        std::vector<std::string> const_variations;
        std::vector<std::string> const_banned_key_tags;
    };

    enum ChoiceEnum
    {
        PLAIN,
        CHOICE,
        CHOICES
    };

    static std::vector<std::string> parse_list_node(const json &j)
    {
        if (j.is_null() || !j.is_object())
            throw std::runtime_error("invalid list node: " + j.dump());

        if (!j.contains("type") || j["type"].get<std::string>() != "list")
            throw std::runtime_error("invalid list node: " + j.dump());

        if (!j.contains("items") || !j["items"].is_array())
            throw std::runtime_error("invalid list node: " + j.dump());

        std::vector<std::string> out;
        out.reserve(j["items"].size());
        for (const auto &el : j["items"])
            out.push_back(el.get<std::string>());

        return out;
    }

    template <typename T>
    static std::string choice(std::vector<T> arr)
    {
        if (arr.size() == 0)
            return T();
        return arr[rng::irange(0, arr.size() - 1)];
    }

    template <typename T>
    static std::vector<T> choices(std::vector<T> arr)
    {
        if (arr.empty())
        {
            return {};
        }

        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_int_distribution<> dis(1, arr.size());
        int x = dis(gen);

        std::vector<T> result;
        result.reserve(x);

        std::sample(arr.begin(), arr.end(), std::back_inserter(result), x, gen);

        return result;
    }

    // parse a reroll node into groups of choices
    static std::vector<std::pair<ChoiceEnum, std::vector<std::string>>> parse_reroll_choice_groups(const json &j)
    {
        std::vector<std::pair<ChoiceEnum, std::vector<std::string>>> groups;
        if (j.is_null())
            return groups;

        const json *root = &j;
        json tmp;
        if (j.is_object() && j.contains("type") && j["type"].get<std::string>() == "list")
        {
            tmp = j.value("items", json::array());
            root = &tmp;
        }

        if (root->is_array())
        {
            for (const auto &el : *root)
            {
                if (el.is_object() && el.contains("type"))
                {
                    const std::string t = el["type"].get<std::string>();
                    if (t == "choice" || t == "choices")
                    {
                        ChoiceEnum c = t == "choice" ? CHOICE : CHOICES;
                        std::vector<std::string> g;
                        for (const auto &s : el["items"])
                            g.push_back(s.get<std::string>());
                        if (!g.empty())
                            groups.push_back({c, std::move(g)});
                        continue;
                    }
                    if (t == "list")
                    {
                        std::vector<std::string> g = parse_list_node(el);
                        if (!g.empty())
                            groups.push_back({PLAIN, std::move(g)});
                        continue;
                    }
                }
            }
        }
        return groups;
    }

    // reroll factory
    static std::function<std::vector<std::string>()> build_reroll_fn(const json &j)
    {
        std::vector<std::pair<ChoiceEnum, std::vector<std::string>>> groups = parse_reroll_choice_groups(j);
        if (groups.empty())
            return nullptr;
        return [groups]() -> std::vector<std::string>
        {
            std::vector<std::string> out;
            out.reserve(groups.size());
            for (const auto &[t, g] : groups)
            {
                if (g.empty())
                    continue;
                switch (t)
                {
                case PLAIN:
                    out.insert(out.end(), g.begin(), g.end());
                    break;
                case CHOICE:
                    out.push_back(choice(g));
                    break;
                case CHOICES:
                    std::vector<std::string> tmp = choices(g);
                    out.insert(out.end(), tmp.begin(), tmp.end());
                    break;
                }
            }
            return out;
        };
    }

    static LoraFields parse_fields(const json &fields)
    {
        LoraFields f;
        f.lora_tag = fields.value("lora_tag", std::string());
        f.prefix_tag = fields.value("prefix_tag", std::string("lora"));
        f.min_weight = fields.value("min_weight", 1.0f);
        f.max_weight = fields.value("max_weight", 1.0f);
        f.const_tags = parse_list_node(fields.value("const_tags", json()));
        f.const_variations = parse_list_node(fields.value("const_variations", json()));
        f.const_banned_key_tags = parse_list_node(fields.value("const_banned_key_tags", json()));
        return f;
    }

} // namespace

Lora_controller *init_lora_controller_from_json(const std::string &json_path)
{
    std::ifstream in(json_path);
    if (!in)
        throw std::runtime_error("Failed to open JSON file: " + json_path);

    json root;
    in >> root;

    const json &instances = root.at("instances");
    std::vector<MainLora *> mains;
    std::vector<SecondaryLora *> secondaries;
    std::vector<ConditionalVariantLora *> conditionals;

    // name -> Lora*
    std::unordered_map<std::string, Lora *> by_name;

    for (const auto &inst : instances)
    {
        const std::string cls = inst.at("class").get<std::string>();
        const json &fields = inst.at("fields");
        const LoraFields f = parse_fields(fields);
        const bool disabled = inst.value("disabled", false);
        const std::string name = inst.at("name").get<std::string>();
        if (disabled)
            continue;

        if (cls == "MainLora")
        {
            auto *obj = new MainLora(f.const_tags, f.const_variations, f.lora_tag, f.const_banned_key_tags, f.min_weight, f.max_weight, f.prefix_tag);
            // attach rerolls
            if (inst.contains("reroll_tags"))
                obj->set_reroll_tags(build_reroll_fn(inst["reroll_tags"]));
            if (inst.contains("reroll_variations"))
                obj->set_reroll_variations(build_reroll_fn(inst["reroll_variations"]));
            mains.push_back(obj);
            by_name[name] = obj;
        }
        else if (cls == "SecondaryLora")
        {
            auto *obj = new SecondaryLora(f.const_tags, f.const_variations, f.lora_tag, f.const_banned_key_tags, f.min_weight, f.max_weight, f.prefix_tag);
            if (inst.contains("reroll_tags"))
                obj->set_reroll_tags(build_reroll_fn(inst["reroll_tags"]));
            if (inst.contains("reroll_variations"))
                obj->set_reroll_variations(build_reroll_fn(inst["reroll_variations"]));
            secondaries.push_back(obj);
            by_name[name] = obj;
        }
        else if (cls == "TagLora")
        {
            auto *obj = new TagLora(f.const_tags, f.const_variations, f.lora_tag, f.const_banned_key_tags, f.min_weight, f.max_weight, f.prefix_tag);
            if (inst.contains("reroll_tags"))
                obj->set_reroll_tags(build_reroll_fn(inst["reroll_tags"]));
            if (inst.contains("reroll_variations"))
                obj->set_reroll_variations(build_reroll_fn(inst["reroll_variations"]));
            // tag loras are stored in tag_lora_map later via tag_lora_map mapping
            by_name[name] = obj;
        }
        else if (cls == "ConditionalVariantLora")
        {
            auto *obj = new ConditionalVariantLora(f.const_tags, f.const_variations, f.lora_tag, f.const_banned_key_tags, f.min_weight, f.max_weight, f.prefix_tag);
            if (inst.contains("extra_ban_categories"))
                obj->register_extra_banned(parse_list_node(inst["extra_ban_categories"]));
            if (inst.contains("reroll_tags"))
                obj->set_reroll_tags(build_reroll_fn(inst["reroll_tags"]));
            if (inst.contains("reroll_variations"))
                obj->set_reroll_variations(build_reroll_fn(inst["reroll_variations"]));
            conditionals.push_back(obj);
            by_name[name] = obj;
        }
    }

    // build tag lora map: json maps text -> key of TagLora name
    std::unordered_map<std::string, TagLora *> tag_lora_map;
    if (root.contains("tag_lora_map"))
    {
        for (auto it = root["tag_lora_map"].begin(); it != root["tag_lora_map"].end(); ++it)
        {
            const std::string key = it.key();
            const std::string mapped = it.value().get<std::string>();
            auto found = by_name.find(mapped);
            if (found != by_name.end())
            {
                TagLora *tagPtr = dynamic_cast<TagLora *>(found->second);
                if (tagPtr)
                    tag_lora_map.emplace(key, tagPtr);
            }
        }
    }

    // build lora compatibility: map of main name -> list of secondaries names
    std::unordered_map<Lora *, std::vector<SecondaryLora *>> compatibility_map;
    if (root.contains("compatibility_map"))
    {
        const json &acm = root["compatibility_map"];
        for (auto it = acm.begin(); it != acm.end(); ++it)
        {
            const std::string actionName = it.key();
            auto actIt = by_name.find(actionName);
            if (actIt == by_name.end())
                continue;
            Lora *actionPtr = actIt->second;
            std::vector<SecondaryLora *> mods;
            for (const auto &modName : it.value())
            {
                auto modIt = by_name.find(modName.get<std::string>());
                if (modIt == by_name.end())
                    continue;
                SecondaryLora *m = dynamic_cast<SecondaryLora *>(modIt->second);
                if (m)
                    mods.push_back(m);
            }
            compatibility_map.emplace(actionPtr, std::move(mods));
        }
    }

    return new Lora_controller(mains, secondaries, conditionals, tag_lora_map, compatibility_map);
}
