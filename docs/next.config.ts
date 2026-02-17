import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/forgetless",
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
