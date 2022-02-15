#!/usr/bin/env bash
mkdir -p results

ids=$(~/ifs_generalization/jhi_calculator_l/azcopy list "$1?$2" | grep extra_iters | cut -d'/' -f1-2)

ids=($ids)

echo $ids

uniqueids=()
while IFS= read -r -d '' x
do
    uniqueids+=("$x")
done < <(printf "%s\0" "${ids[@]}" | sort -uz)

for d in "${uniqueids[@]}"; do

  b="$1/$d?$2"

  echo "In folder $d"

  IFS='/' read -r -a upper_folder <<< "$d"

  echo "./results/${upper_folder[1]}_dom_eig.pkl"

  if [[ ! -f "./results/${upper_folder[1]}_dom_eig.pkl" ]]; then

    ~/ifs_generalization/jhi_calculator_l/azcopy cp $b ./data  --recursive

    for sd in ./data/*; do

      if [ -d "$sd/extra_iters" ]; then
          if [[ $sd =~ "cifar10" ]]; then
              dataset="cifar10"
          elif [[ $sd =~ "cifar100" ]]; then
              dataset="cifar100"
          elif [[ $sd =~ "mnist" ]]; then
              dataset="mnist"
          elif [[ $sd =~ "svhn" ]]; then
                dataset="svhn"
          fi

          IFS='/' read -r -a folders <<< "$sd"
          IFS='_' read -r -a array <<< "${folders[2]}"
          lr=${array[${#array[@]} - 1]}
          bs=${array[${#array[@]} - 2]}


            jhi -f $sd/extra_iters -b $bs -l $lr -d $dataset -m 50 -g $3 -s ./results -e $4

      fi
    done
  else
    echo "Already have results for $d"
  fi
  rm -r ./data
done
