#linux删除文件夹
（1）rm -rf 目录名称

    -r 向下递归，不管有多少级目录，一并删除
    -f 直接强行删除，没有任何提示
删除文件夹实例：

	rm -rf /var/log/httpd #删除/var/log/httpd目录以及其下所有文件、文件夹
    rm -f /var/log/httpd/access.log  #q强制删除access.log 这个文件
注意：
在linux中是没有设置回收站，因此在使用rm命令的时候一定要小心些，删除后的文件是无法恢复的
