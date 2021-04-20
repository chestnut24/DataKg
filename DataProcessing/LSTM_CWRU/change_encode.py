# 批量更改csv文件编码为utf-8
# i!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import sys
import codecs
import chardet


def convert(filename, out_enc="UTF-8"):
    try:
        content = codecs.open(filename, 'rb').read()
        source_encoding = chardet.detect(content)['encoding']
        print("file encoding:%s" % source_encoding)

        if source_encoding != None:
            content = content.decode(source_encoding).encode(out_enc)
            codecs.open(filename, 'wb').write(content)
            print(filename, "更改为utf-8")
            # content.close()
        else:
            print("can not recgonize file encoding %s" % filename)

    except IOError as err:
        print("I/O error:{0}".format(err))


def explore(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                print("filename:%s" % file)
                path = os.path.join(root, file)
                convert(path)


def main():
    # explore(os.getcwd())  # 当前文件夹
    filepath = r"../.././import"  # 指定路径
    explore(filepath)


if __name__ == "__main__":
    main()
