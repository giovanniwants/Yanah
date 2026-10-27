#!/bin/bash
cd "/Users/admin/Desktop/yanah_build"
source .venv/bin/activate

# 1. Start the MLX engine server on 8888 (replaces Sushi) if it's not already running
if ! curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8888/health 2>/dev/null | grep -q 200; then
  echo "==> starting mlx_server on 8888"
  python mlx_server.py > ~/mlx-server.log 2>&1 &

  echo "    loading the model (about 1-3 min)"
  for i in $(seq 1 240); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8888/health 2>/dev/null)
    if [ "$code" = "200" ]; then
      echo "    mlx_server up"
      break
    fi
    if [ "$code" = "500" ]; then
      echo "    mlx_server failed to load - see ~/mlx-server.log"
      break
    fi
    sleep 1
  done
fi

# 2. Start Yanah daemon
echo "==> starting yanah daemon on 8901"
pkill -9 -f yanah_shell.py 2>/dev/null
sleep 1
export YANAH_BASE_URL=http://127.0.0.1:8888/v1
python3 yanah_shell.py > ~/yanah.log 2>&1 &

for i in $(seq 1 10); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8901/health 2>/dev/null | grep -q 200; then
    echo "    yanah up"
    break
  fi
  sleep 1
done

echo "==> opening Yanah"
open -a "Google Chrome" "/Users/admin/Desktop/yanah_build/index.html"
echo "Yanah is running."
