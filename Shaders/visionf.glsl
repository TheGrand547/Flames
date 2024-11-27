#version 440 core

layout(location = 0) in vec2 fTex;
layout(location = 1) in float depth;
layout(location = 2) flat in vec3 fPos;

layout(location = 0) out vec4 fColor;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

uniform float radius;
uniform int featureToggle;

const float PI = 1.0 / radians(180);

void main()
{
	// From https://github.com/paroj/gltut/blob/master/Tut%2013%20Impostors/data/GeomImpostor.frag
	vec3 adjusted = vec3(fTex * radius, 0.0) + fPos;
	vec3 ray = normalize(adjusted);
	
	float B = 2.0 * dot(ray, -fPos);
	float C = dot(fPos, fPos) - (radius * radius);
	
	float det = (B * B) - (4 * C);
	if(det < 0.0)
		discard;
		
	float sqrtDet = sqrt(det);
	float posT = (-B + sqrtDet)/2;
	float negT = (-B - sqrtDet)/2;
	
	// TODO: To get Far simply replace min with max
	float intersectT = min(posT, negT);
	
	// Outputs
	vec3 finalPos = ray * intersectT;
	vec3 finalNorm = normalize(finalPos - fPos);
	
	// TODO: Figure out how to avoid matrix multiplication in fragment shader if at all possible
	vec4 clipPos = Projection * vec4(finalPos, 1.0);
	float ndcDepth = clipPos.z / clipPos.w;
	gl_FragDepth = ((gl_DepthRange.diff * ndcDepth) + gl_DepthRange.near + gl_DepthRange.far) / 2.0;
	fColor = vec4(1, 0, 0.5, 1);
}