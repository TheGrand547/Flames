#pragma once
#ifndef GL_UTIL_H
#define GL_UTIL_H
#include <glew.h>
#include <type_traits>

enum FramebufferClearFlags : int
{
    ColorBuffer   = 1 << 0, 
    DepthBuffer   = 1 << 1,
    StencilBuffer = 1 << 2
};

enum GLFeatureFlags : int
{
    Blending            = 1 << 0,  // See glBlendFunc
    ColorLogicOperation = 1 << 1,  // See glLogicOp
    FaceCulling         = 1 << 2,  // See glCullFace
    DebugOutput         = 1 << 3,  // Enables explicit debug outputs via callback
    DepthClamp          = 1 << 4,  // See glDepthRange
    DepthTesting        = 1 << 5,  // See glDepthTest and glDepthRange
    Dither              = 1 << 6,  // Think this is deprecated/doesn't work
    FramebufferSRGB     = 1 << 7,  // Requires a sRGB framebuffer
    SmoothLines         = 1 << 8,  // See glLineWidth
    MultiSampling       = 1 << 9,  // See glSampleCoverage
    PolygonOffsetFill   = 1 << 10, // See glPolygonOffset
    PolygonOffsetLine   = 1 << 11, // See glPolygonOffset
    PolygonOffsetPoint  = 1 << 12, // See glPolygonOffset
    SmoothPolgyons      = 1 << 13, // Smoothes polygon edges somehow
    PrimitiveRestarting = 1 << 14, // See glPrimitiveRestartIndex
    RasterizerDiscard   = 1 << 15, // Primitives are discarded before rasterization
    SampleAlphaCoverage = 1 << 16, // See glSampleCoverage
    SampleAlphaToOne    = 1 << 17, // See glSampleCoverage
    SampleCoverage      = 1 << 18, // See glSampleCoverage
    SampleShading       = 1 << 19, // See glMinSampleShading
    SampleMask          = 1 << 20, // See glSampleMaski
    ScissorTesting      = 1 << 21, // See glScissor
    StencilTesting      = 1 << 22, // See glStencilFunc and glStencilOp
    SeamlessCubeMap     = 1 << 23, // Proper blending on cubemap side intersections
    ShaderPointSize     = 1 << 24, // Shader defined gl_PointSize will not be ignored
};

consteval FramebufferClearFlags operator|(FramebufferClearFlags lhs, FramebufferClearFlags rhs) {
    return static_cast<FramebufferClearFlags>(
        static_cast<std::underlying_type<FramebufferClearFlags>::type>(lhs) |
        static_cast<std::underlying_type<FramebufferClearFlags>::type>(rhs)
        );
}

consteval GLFeatureFlags operator|(GLFeatureFlags lhs, GLFeatureFlags rhs) {
    return static_cast<GLFeatureFlags>(
        static_cast<std::underlying_type<GLFeatureFlags>::type>(lhs) |
        static_cast<std::underlying_type<GLFeatureFlags>::type>(rhs)
        );
}

template<FramebufferClearFlags flags = ColorBuffer>
inline void ClearFramebuffer()
{
    GLenum flagged = 0;
    if constexpr ((flags & ColorBuffer))
    {
        flagged |= GL_COLOR_BUFFER_BIT;
    }
    if constexpr ( (flags & DepthBuffer))
    {
        flagged |= GL_DEPTH_BUFFER_BIT;
    }
    if constexpr( (flags & StencilBuffer))
    {
        flagged |= GL_STENCIL_BUFFER_BIT;
    }
    glClear(flagged);
}

typedef void(*GLEnableFunc)(GLenum);
template<GLEnableFunc func, GLFeatureFlags flags>
inline void ApplyFeatureFlags()
{
    if constexpr (flags & Blending)            func(GL_BLEND);
    if constexpr (flags & ColorLogicOperation) func(GL_COLOR_LOGIC_OP);
    if constexpr (flags & FaceCulling)         func(GL_CULL_FACE);
    if constexpr (flags & DebugOutput)         func(GL_DEBUG_OUTPUT);
    if constexpr (flags & DepthClamp)          func(GL_DEPTH_CLAMP);
    if constexpr (flags & DepthTesting)        func(GL_DEPTH_TEST);
    if constexpr (flags & Dither)              func(GL_DITHER);
    if constexpr (flags & FramebufferSRGB)     func(GL_FRAMEBUFFER_SRGB);
    if constexpr (flags & SmoothLines)         func(GL_LINE_SMOOTH);
    if constexpr (flags & MultiSampling)       func(GL_MULTISAMPLE);
    if constexpr (flags & PolygonOffsetFill)   func(GL_POLYGON_OFFSET_FILL);
    if constexpr (flags & PolygonOffsetLine)   func(GL_POLYGON_OFFSET_LINE);
    if constexpr (flags & PolygonOffsetPoint)  func(GL_POLYGON_OFFSET_POINT);
    if constexpr (flags & SmoothPolgyons)      func(GL_POLYGON_SMOOTH);
    if constexpr (flags & PrimitiveRestarting) func(GL_PRIMITIVE_RESTART);
    if constexpr (flags & RasterizerDiscard)   func(GL_RASTERIZER_DISCARD);
    if constexpr (flags & SampleAlphaCoverage) func(GL_SAMPLE_ALPHA_TO_COVERAGE);
    if constexpr (flags & SampleAlphaToOne)    func(GL_SAMPLE_ALPHA_TO_ONE);
    if constexpr (flags & SampleCoverage)      func(GL_SAMPLE_COVERAGE);
    if constexpr (flags & SampleShading)       func(GL_SAMPLE_SHADING);
    if constexpr (flags & SampleMask)          func(GL_SAMPLE_MASK);
    if constexpr (flags & ScissorTesting)      func(GL_SCISSOR_TEST);
    if constexpr (flags & StencilTesting)      func(GL_STENCIL_TEST);
    if constexpr (flags & SeamlessCubeMap)     func(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    if constexpr (flags & ShaderPointSize)     func(GL_PROGRAM_POINT_SIZE);
}

template<GLFeatureFlags flags>
inline void EnableGLFeatures()
{
    ApplyFeatureFlags<glEnable, flags>();
}

template<GLFeatureFlags flags>
inline void DisableGLFeatures()
{
    ApplyFeatureFlags<glDisable, flags>();
}

inline void BindDefaultFrameBuffer()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

inline void DisableDepthBufferWrite()
{
    glDepthMask(GL_FALSE);
}

inline void EnableDepthBufferWrite()
{
    glDepthMask(GL_TRUE);
}

template<GLFeatureFlags flags, bool enable = true> struct FeatureFlagPush
{
    inline FeatureFlagPush() noexcept
    {
        if constexpr (enable)
        {
            EnableGLFeatures<flags>();
        }
        else
        {
            DisableGLFeatures<flags>();
        }
    }

    inline ~FeatureFlagPush() noexcept
    {
        if constexpr (enable)
        {
            DisableGLFeatures<flags>();
        }
        else
        {
            EnableGLFeatures<flags>();
        }
    }
};

#define __GL_FLAG_PUSH(x, y, z) FeatureFlagPush<x, z> __##y##{}
#define __GL_FLAG_PUSH2(x, y, z) __GL_FLAG_PUSH(x, y, z)
#define EnablePushFlags(x) __GL_FLAG_PUSH2(x, __LINE__, true)
#define DisablePushFlags(x) __GL_FLAG_PUSH2(x, __LINE__, false)

#endif // GL_UTIL_H