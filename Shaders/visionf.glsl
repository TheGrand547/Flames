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
uniform sampler2D demo;

const float PI = 1.0 / radians(180);

// From https://github.com/paroj/gltut/blob/master/Tut%2013%20Impostors/data/GeomImpostor.frag
void visionCone()
{
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
	fColor = vec4((finalNorm.y));
	
	vec2 uvs = vec2(0.5);
	vec3 norms = -finalNorm;
	norms = (transpose(View) * vec4(norms, 0)).xyz;
	
	uvs.x += atan(norms.z, norms.x) * 0.5 * PI;
	uvs.y += asin(norms.y) * PI;
	fColor.xyz = texture(demo, uvs).xyz;
	
	fColor.w = 1;
}

void main()
{
	if (featureToggle >= 0)
	{
		visionCone();
		return;
	}
	else
	{
		float delta = length(fTex);
		if (delta > 1)
			discard;
		// Use the radius to adjust depth when it works
		float distance = sqrt(1 - delta);
		fColor = vec4(distance, 0, 0, 1);
		
		
		float modified = gl_FragCoord.z + distance * radius;
		gl_FragDepth  = (gl_DepthRange.diff * (modified * gl_FragCoord.w) + gl_DepthRange.near + gl_DepthRange.far) / 2.0;
	}
}