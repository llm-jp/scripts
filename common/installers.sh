install_python() {
  # This function installs CPython.
  # Usage: install_python BRANCH_TAG TARGET_DIR
  # Arguments:
  #  - BRANCH_TAG: Branch or tag of CPython to be checked out.
  #  - TARGET_DIR: Directory where CPython will be installed. 

  if [ $# -ne 2 ]; then
    set +x
    >&2 echo Usage: install_python BRANCH_TAG TARGET_DIR
    exit 1
  fi

  git clone https://github.com/python/cpython -b $1
  pushd cpython
  ./configure --prefix=$2 --enable-optimizations
  make -j 64
  make install
  popd 
}

install_python() {
  # This function installs CPython.
  # Usage: install_python BRANCH_TAG TARGET_DIR
  # Arguments:
  #  - BRANCH_TAG: Branch or tag of CPython to be checked out.
  #  - TARGET_DIR: Directory where CPython will be installed. 

  if [ $# -ne 2 ]; then
    set +x
    >&2 echo Usage: install_python BRANCH_TAG TARGET_DIR
    exit 1
  fi
}
