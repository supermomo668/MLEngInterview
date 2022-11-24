# using ubuntu LTS version
FROM ubuntu:20.04 AS builder

# source installation
# RUN apt update && apt upgrade -y && apt install wget -y
# RUN wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz && tar -xf Python-3.7.4.tgz && mv Python-3.7.4 /opt/ && apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev -y & ldconfig /opt/Python3.7.4

# WORKDIR /application

RUN apt-get update && apt-get install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install --no-install-recommends -y python3.7 python3.7-venv python3-pip python3-wheel build-essential -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# RUN cd /opt/Python-3.7.4/ && ./configure --enable-optimizations --enable-shared && make && make -j 4 && sudo make altinstall

#

# create and activate virtual environment
RUN pip3 install virtualenv && virtualenv env && . env/bin/activate
ENV PATH="env/bin:$PATH"

# install requirements
COPY . /
RUN pip3 install --no-cache-dir -r requirements.txt

FROM ubuntu:20.04 AS runner
# RUN apt-get update && apt-get install --no-install-recommends -y python3.7 python3-venv && apt-get clean && rm -rf /var/lib/apt/lists/*


COPY --from=builder . /

# activate virtual environment
ENV VIRTUAL_ENV=/env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["tensorboard"]
#CMD ["python"]
ENTRYPOINT ["python","main.py"]