#!/usr/bin/env bash
mkdir -p results

for d in $1/*/; do

  a=$(basename $d)

  if [[ $a =~ "cifar10"  ||  $a =~ "cifar100"  ||  $a =~ "mnist" ||  $b =~ "svhn"]]; then
    for sd in $d/*/; do
      b=$(basename $sd)

      echo "In folder $b"
      if [[ $b =~ "cifar10"  ||  $b =~ "cifar100"  ||  $b =~ "mnist" ||  $b =~ "svhn" ]]; then

        if [ -d "$sd/extra_iters" ]; then
          if [[ $b =~ "cifar10" ]]; then
            dataset="cifar10"
          elif [[ $b =~ "cifar100" ]]; then
            dataset="cifar100"
          elif [[ $b =~ "mnist" ]]; then
            dataset="mnist"
          elif [[ $b =~ "svhn" ]]; then
            dataset="svhn"
          fi

          IFS='_' read -r -a array <<< "$b"
          lr=${array[${#array[@]} - 1]}
          bs=${array[${#array[@]} - 2]}

          jhi -f $sd/extra_iters -b $bs -l $lr -d $dataset -m 50 -g $2 -s ./results -e $3
        fi

      fi
    done
  fi
done
