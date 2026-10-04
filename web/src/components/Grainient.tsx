import { useEffect, useRef } from "react";

/**
 * Grainient - an atmospheric WebGL gradient layer.
 *
 * Domain-warped value noise rendered in a fragment shader, tuned for a
 * restrained industrial palette. The canvas is decorative: it carries no
 * information, so it is aria-hidden and pauses whenever it is offscreen,
 * the tab is hidden, or the visitor prefers reduced motion (a single
 * static frame is drawn instead). If WebGL is unavailable the component
 * renders nothing and the parent's static gradient shows through.
 */

export interface GrainientParams {
  /** deep base color, linear-ish 0..1 */
  colorA: [number, number, number];
  /** lifted surface color */
  colorB: [number, number, number];
  /** energy accent (cyan) */
  colorC: [number, number, number];
  /** secondary signal (green), used sparingly */
  colorD: [number, number, number];
  noiseScale: number;
  warpAmp: number;
  /** brightness of the accent band, 0..1 */
  energy: number;
  grain: number;
  vignette: number;
  speed: number;
}

export const INDUSTRIAL_PARAMS: GrainientParams = {
  colorA: [0.008, 0.022, 0.013],
  colorB: [0.03, 0.09, 0.05],
  colorC: [0.16, 0.55, 0.3],
  colorD: [0.1, 0.45, 0.42],
  noiseScale: 2.1,
  warpAmp: 0.55,
  energy: 0.6,
  grain: 0.028,
  vignette: 0.8,
  speed: 0.045,
};

const VERT = `#version 300 es
in vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;

uniform vec2 uResolution;
uniform float uTime;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform vec3 uColorC;
uniform vec3 uColorD;
uniform float uNoiseScale;
uniform float uWarpAmp;
uniform float uEnergy;
uniform float uGrain;
uniform float uVignette;
out vec4 outColor;

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
  float v = 0.0;
  float amp = 0.5;
  mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
  for (int i = 0; i < 5; i++) {
    v += amp * vnoise(p);
    p = rot * p * 2.03;
    amp *= 0.5;
  }
  return v;
}

void main() {
  vec2 uv = gl_FragCoord.xy / uResolution.xy;
  vec2 p = uv - 0.5;
  p.x *= uResolution.x / uResolution.y;
  p *= uNoiseScale;

  float t = uTime;

  // two-pass domain warp (iq style)
  vec2 q = vec2(fbm(p + vec2(0.0, 0.0) + t * 0.5),
                fbm(p + vec2(5.2, 1.3) - t * 0.35));
  vec2 r = vec2(fbm(p + uWarpAmp * q + vec2(1.7, 9.2) + t * 0.28),
                fbm(p + uWarpAmp * q + vec2(8.3, 2.8) - t * 0.22));
  float f = fbm(p + uWarpAmp * r);

  // graphite foundation
  vec3 col = mix(uColorA, uColorB, smoothstep(0.15, 0.85, f));

  // restrained cyan energy current where the field folds
  float fold = smoothstep(0.42, 0.5, abs(r.x - 0.5));
  float current = (1.0 - fold) * smoothstep(0.25, 0.75, f);
  col += uColorC * current * uEnergy * 0.55;

  // sparse green signal, fainter still
  float signal = smoothstep(0.48, 0.5, abs(q.y - 0.47)) * smoothstep(0.3, 0.8, f);
  col += uColorD * (1.0 - signal) * uEnergy * 0.16;

  // keep the field mostly dark: lift highlights only slightly
  col = pow(max(col, vec3(0.0)), vec3(1.1));

  // vignette so text areas stay deep
  float vig = 1.0 - uVignette * dot(uv - 0.5, uv - 0.5) * 1.9;
  col *= clamp(vig, 0.0, 1.0);

  // film grain
  float g = hash(gl_FragCoord.xy + fract(uTime) * 100.0) - 0.5;
  col += g * uGrain;

  outColor = vec4(col, 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string) {
  const sh = gl.createShader(type);
  if (!sh) return null;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

export default function Grainient({
  params = INDUSTRIAL_PARAMS,
  className,
}: {
  params?: GrainientParams;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
    if (!gl) return;

    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) return;
    const prog = gl.createProgram();
    if (!prog) return;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW,
    );
    const loc = gl.getAttribLocation(prog, "position");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    const U = (name: string) => gl.getUniformLocation(prog, name);
    const uResolution = U("uResolution");
    const uTime = U("uTime");
    const uColorA = U("uColorA");
    const uColorB = U("uColorB");
    const uColorC = U("uColorC");
    const uColorD = U("uColorD");
    const uNoiseScale = U("uNoiseScale");
    const uWarpAmp = U("uWarpAmp");
    const uEnergy = U("uEnergy");
    const uGrain = U("uGrain");
    const uVignette = U("uVignette");

    gl.uniform3fv(uColorA, params.colorA);
    gl.uniform3fv(uColorB, params.colorB);
    gl.uniform3fv(uColorC, params.colorC);
    gl.uniform3fv(uColorD, params.colorD);
    gl.uniform1f(uNoiseScale, params.noiseScale);
    gl.uniform1f(uWarpAmp, params.warpAmp);
    gl.uniform1f(uEnergy, params.energy);
    gl.uniform1f(uGrain, params.grain);
    gl.uniform1f(uVignette, params.vignette);

    // render at a capped device pixel ratio: this layer is soft gradients,
    // extra pixels buy nothing
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    let running = false;
    let raf = 0;
    let visible = true;
    const start = performance.now();

    const resize = () => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (w === 0 || h === 0) return false;
      const pw = Math.round(w * dpr);
      const ph = Math.round(h * dpr);
      if (canvas.width !== pw || canvas.height !== ph) {
        canvas.width = pw;
        canvas.height = ph;
        gl.viewport(0, 0, pw, ph);
      }
      gl.uniform2f(uResolution, canvas.width, canvas.height);
      return true;
    };

    const draw = (now: number) => {
      if (!resize()) return;
      gl.uniform1f(uTime, ((now - start) / 1000) * params.speed);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const loop = () => {
      if (!running || !visible) return;
      draw(performance.now());
      raf = requestAnimationFrame(loop);
    };

    const play = () => {
      if (reduced || !running || !visible) return;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(loop);
    };

    // draw one frame regardless; with reduced motion this is the only frame
    running = true;
    draw(performance.now());
    if (!reduced) play();

    const io = new IntersectionObserver(
      (entries) => {
        const entry = entries[entries.length - 1];
        if (!entry) return;
        visible = entry.isIntersecting;
        if (visible) play();
        else cancelAnimationFrame(raf);
      },
      { threshold: 0 },
    );
    io.observe(canvas);

    const onVisibility = () => {
      visible = !document.hidden;
      if (visible) play();
      else cancelAnimationFrame(raf);
    };
    document.addEventListener("visibilitychange", onVisibility);

    const ro = new ResizeObserver(() => {
      if (reduced || !visible) draw(performance.now());
    });
    ro.observe(canvas);

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      io.disconnect();
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      gl.deleteProgram(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteBuffer(buf);
    };
  }, [params]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      aria-hidden="true"
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
    />
  );
}
