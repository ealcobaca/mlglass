#!/bin/bash

create() {
    cat > pbs_$2.sh << END
    module load python/3.5.4
    cd $1
    source ../../env3.5/bin/activate
    python run.py -e 4 -i ${2}
  
END
  chmod a+x pbs_$2.sh
}

delete() {
  rm pbs_$1.sh
}

super() {

    echo $2
    nick="exp-${2}"

    create $1 $2
    qsub -N $nick -l select=1:ncpus=$3:mem=$4gb,walltime=$5:00:00 pbs_$2.sh
    sleep 5
    delete $2
}


mem=10
cpus=10
cd $1 

for i in `seq 0 362`; do
    while true; do

        SHT=$(totaljob | grep Parallel_short | sed "s/\s\+/|/g" | cut -d"|" -f5)
        LNG=$(totaljob | grep Parallel_long  | sed "s/\s\+/|/g" | cut -d"|" -f5)
        RTQ=$(totaljob | grep RouteQ | sed "s/\s\+/|/g" | cut -d"|" -f5)

        if [ "$LNG" -ge $cpus ] && [ "$RTQ" -gt 0 ]; then
            super $1 ${i} $cpus $mem 336
            sleep 5
            break
        elif [ "$SHT" -ge $cpus ] && [ "$RTQ" -gt 0 ]; then
            super $1 $2 ${i} $cpus $mem 24
            sleep 5
            break
        else
            echo "sleeping..."
            sleep 60
        fi
    done
done
