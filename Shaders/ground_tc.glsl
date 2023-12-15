#version 440 core

layout (vertices=4) out;
layout(location = 0) in vec3 tcPos[];
layout(location = 1) in vec3 tcNorm[];
layout(location = 2) in vec2 tcTex[];

layout(location = 0) out vec3 tePos[];
layout(location = 1) out vec3 teNorm[];
layout(location = 2) out vec2 teTex[];

uniform int amount;

void main()
{
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    teTex[gl_InvocationID] = tcTex[gl_InvocationID];
	teNorm[gl_InvocationID] = tcNorm[gl_InvocationID];
	tePos[gl_InvocationID] = tcPos[gl_InvocationID];
    if (gl_InvocationID == 0)
    {
        gl_TessLevelOuter[0] = amount;
        gl_TessLevelOuter[1] = amount;
        gl_TessLevelOuter[2] = amount;
        gl_TessLevelOuter[3] = amount;

        gl_TessLevelInner[0] = amount;
        gl_TessLevelInner[1] = amount;
    }
}