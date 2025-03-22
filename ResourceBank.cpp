#include "ResourceBank.h"

namespace ShaderBank
{
    static std::unordered_map<std::string, Shader> stored;

    Shader& Get(const std::string& name)
    {
        return stored[name];
    }
}

namespace VAOBank
{
    static std::unordered_map<std::string, VAO> stored;

    VAO& Get(const std::string& name)
    {
        return stored[name];
    }
}
