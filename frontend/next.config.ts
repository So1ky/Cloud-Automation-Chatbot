import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 컨테이너 배포용 최소 번들 출력 (dev 동작에는 영향 없음)
  output: "standalone",
};

export default nextConfig;
