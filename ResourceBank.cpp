#include "ResourceBank.h"

namespace ShaderBank
{
    static std::unordered_map<std::string_view, Shader> stored;

    Shader& Get(const std::string_view& name)
    {
        return stored[name];
    }
}

namespace VAOBank
{
    static std::unordered_map<std::string_view, VAO> stored;

    VAO& Get(const std::string_view& name)
    {
        return stored[name];
    }
}

namespace BufferBank
{
    static std::unordered_map<std::string_view, ArrayBuffer> stored;

    ArrayBuffer& BufferBank::Get(const std::string_view& name)
    {
        return stored[name];
    }
}
