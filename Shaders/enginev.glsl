#version 440 core

layout(location = 0) in vec4 Position;

layout(location = 0) out vec4 colorIn;

layout(std140, binding = 0) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

// TODO: uniform
uniform uint Time;
uniform uint Period;

// Instanced shader expecting 

vec3 Points[] = 
{
	vec3( 0.5,    0, -sqrt(2.f) / 4.f),
	vec3(-0.5,    0, -sqrt(2.f) / 4.f),
	vec3(   0,  0.5,  sqrt(2.f) / 4.f),
	vec3(   0, -0.5,  sqrt(2.f) / 4.f)
};

vec3 Colors[] = 
{
	vec3(1, 0, 0), vec3(1, 1, 0), vec3(1, 0.65, 0), vec3(1, 0.4, 0)
};

const uint IndexCount = 12;

uint Index[IndexCount] = {
    0, 1, 2,
    0, 2, 3,
    0, 3, 1,
    1, 3, 2
};

// 2D rotation formula, in column order
mat2 rotate(float angle)
{
	float cosine = cos(angle);
 	float sine = sin(angle);
 	return mat2(cosine, sine, 
				-sine, cosine);
}

#define PI (3.1415926538)
#define TAU (2 * PI)
#define OFFSET (PI * 0.13554)

void main()
{
	uint index = gl_VertexID % IndexCount;
	uint indexed = Index[index];
	float ratio = (TAU) * float(Time) / float(Period);
	// Provide variance between the three individual tetrahedrons
	float instanceOffset = floor(gl_VertexID / 12);
	float offset = gl_InstanceID * (OFFSET + instanceOffset) + instanceOffset;
	colorIn = vec4(Colors[indexed], 1);
	
	vec3 local = Points[indexed] * Position.w * 0.5f;
	
	// Rotate around z axis
	local.xy *= rotate(1.3 * ratio + offset);
	// Rotate around x axis
	local.yz *= rotate(0.843 * ratio - offset);
	// Rotate about y axis
	local.xz *= rotate(0.235 * ratio + offset);
	
	vec3 absolute = local + Position.xyz;

	gl_Position = Projection * View * vec4(absolute, 1.0);
}