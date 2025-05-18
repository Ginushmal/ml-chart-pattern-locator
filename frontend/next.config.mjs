/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export", // Add this line
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
