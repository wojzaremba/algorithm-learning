#!/usr/bin/python
import os
import sys

assert(len(sys.argv) <= 2)
prefix = "addition"
if len(sys.argv) == 2: 
  prefix = sys.argv[1]

def getch():
  import sys, tty, termios
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
      tty.setraw(sys.stdin.fileno())
      ch = sys.stdin.read(1)
  finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
  return ch

os.system("cp movie/* /tmp")


content = []
files = os.listdir('movie')
end = 1
for f in files:
  if len(f) <= len(prefix) + 1:
    continue
  try:
    if f[:len(prefix)] == prefix:
      end = max(end, int(f[(len(prefix) + 1):]))
  except:
    pass

for i in range(end):
  with open('movie/%s_%d' % (prefix, i + 1), 'r') as content_file:
    content.append(content_file.read())

current = 0

while True:
  for i in range(200):
    print("")
  print("%d / %d" % (current + 1, len(content)))
  print(content[current])
  print("\nPress ``s'' to show next frame, ``a'' the previous frame, and ``q'' to exit.")
  ch = getch()
  if ch == 'a':
    current = (current - 1) % len(content)
  elif ch == 's':
    current = (current + 1) % len(content)
  elif ch == 'q':
    exit(0)
  
