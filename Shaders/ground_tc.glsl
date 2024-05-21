#version 440 core

layout (vertices=4) out;
layout(location = 0) in vec3 tcPos[];
layout(location = 1) in vec3 tcNorm[];
layout(location = 2) in vec2 tcTex[];

layout(location = 0) out vec3 tePos[];
layout(location = 1) out vec3 teNorm[];
layout(location = 2) out vec2 teTex[];

uniform int amount;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    teTex[gl_InvocationID] = tcTex[gl_InvocationID];
	teNorm[gl_InvocationID] = tcNorm[gl_InvocationID];
	tePos[gl_InvocationID] = tcPos[gl_InvocationID];
    if (gl_InvocationID == 0)
    {
		const int minTesselation = 1;
		const int maxTesselation = 10;
		const float minDistance = 1;
		const float maxDistance = 80;
		const float distanceDelta = maxDistance - minDistance;
	
		float depth0 = (View * gl_in[0].gl_Position).z;
		float depth1 = (View * gl_in[1].gl_Position).z;
		float depth2 = (View * gl_in[2].gl_Position).z;
		float depth3 = (View * gl_in[3].gl_Position).z;

		float delta0 = clamp((abs(depth0) - minDistance) / distanceDelta, 0, 1);
		float delta1 = clamp((abs(depth1) - minDistance) / distanceDelta, 0, 1);
		float delta2 = clamp((abs(depth2) - minDistance) / distanceDelta, 0, 1);
		float delta3 = clamp((abs(depth3) - minDistance) / distanceDelta, 0, 1);
		
		
        gl_TessLevelOuter[0] = mix(maxTesselation, minTesselation, min(delta0, delta1));
        gl_TessLevelOuter[1] = mix(maxTesselation, minTesselation, min(delta1, delta2));
        gl_TessLevelOuter[2] = mix(maxTesselation, minTesselation, min(delta2, delta3));
        gl_TessLevelOuter[3] = mix(maxTesselation, minTesselation, min(delta0, delta3));

        gl_TessLevelInner[0] = max(gl_TessLevelOuter[0], gl_TessLevelOuter[1]);
        gl_TessLevelInner[1] = max(gl_TessLevelOuter[2], gl_TessLevelOuter[3]);
    }
}