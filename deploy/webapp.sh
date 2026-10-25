#!/bin/bash
# 桑叶病虫害检测系统(树莓派端)网页服务 —— 启动/停止/状态
# 用法:  ./webapp.sh start|stop|restart|status|log
cd "$(dirname "$0")" || exit 1
PORT=7860
LOG=app.log
PY=./venv/bin/python
if [ ! -x "$PY" ] && command -v python3 >/dev/null 2>&1; then PY=python3; fi

ip() { hostname -I | awk '{print $1}'; }

case "$1" in
  start)
    if pgrep -f "pi_app.py" >/dev/null; then
      echo "已在运行: http://$(ip):$PORT"
      exit 0
    fi
    setsid nohup $PY pi_app.py --port $PORT > $LOG 2>&1 < /dev/null &
    sleep 8
    if pgrep -f "pi_app.py" >/dev/null; then
      echo "启动成功:"
      echo "  本机访问   http://127.0.0.1:$PORT"
      echo "  局域网访问 http://$(ip):$PORT"
    else
      echo "启动失败, 日志末尾:"
      tail -15 $LOG
    fi
    ;;
  stop)
    pkill -f "pi_app.py" && echo "已停止" || echo "未在运行"
    ;;
  restart)
    $0 stop; sleep 2; $0 start
    ;;
  status)
    if pgrep -f "pi_app.py" >/dev/null; then
      echo "运行中: http://$(ip):$PORT   (pid $(pgrep -f pi_app.py | head -1))"
    else
      echo "未运行"
    fi
    ;;
  log)
    tail -30 $LOG
    ;;
  *)
    echo "用法: $0 start|stop|restart|status|log"
    ;;
esac
