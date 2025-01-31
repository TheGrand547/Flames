#version 440 core

layout(location = 0) in vec3 vPos;

layout(location = 0) out vec3 fTex;

layout(std140, binding = 0) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
	fTex = vPos;
	
	vec4 temp = Projection * vec4(mat3(View) * vPos, 1);
	gl_Position = temp.xyww;
}