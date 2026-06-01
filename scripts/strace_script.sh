strace -p 305963 -c -o strace_out.txt &
sleep 5
kill $!
cat strace_out.txt
