#version 440 core

layout (vertices=3) out;
layout(location = 0) in vec3 fPos[];
layout(location = 1) in vec3 fNorm[];
layout(location = 2) in vec2 fTex[];

layout(location = 0) out vec3 fPos2[];
layout(location = 1) out vec3 fNorm2[];
layout(location = 2) out vec2 fTex2[];

void main()
{
   
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    fTex2[gl_InvocationID] = fTex[gl_InvocationID];
	fNorm2[gl_InvocationID] = fNorm[gl_InvocationID];
	fPos2[gl_InvocationID] = fPos[gl_InvocationID];
    if (gl_InvocationID == 0)
    {
        gl_TessLevelOuter[0] = 16;
        gl_TessLevelOuter[1] = 16;
        gl_TessLevelOuter[2] = 16;

        gl_TessLevelInner[0] = 16;
        gl_TessLevelInner[1] = 16;
    }
}