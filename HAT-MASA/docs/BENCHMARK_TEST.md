# Benchmark Test

Here, we provide the main experimental results and instructions on how to run them. 
**Overall, we use the model provided by [MASA](https://github.com/siyuanliii/masa) to extract target features and complete tracking with our [modified tracker](../masa/models/tracker/masa_tao_transfer_tracker.py).**

## Main Results

<table border="1">
  <tr>
    <th rowspan="2">Methods</th>
    <th colspan="2">DanceTrack test</th>
    <th colspan="2">SportsMOT test</th>
    <th colspan="2">TAO val</th>
  </tr>
  <tr>
    <td>HOTA</td>
    <td>AssA</td>
    <td>IDF1</td>
    <td>HOTA</td>
    <td>AssA</td>
    <td>IDF1</td>
    <td>TETA</td>
    <td>AssocA</td>
  </tr>
  <tr>
    <td>MASA-R50</td>
    <td>50.8</td>
    <td>31.6</td>
    <td>48.8</td>
    <td>71.6</td>
    <td>58.9</td>
    <td>71.7</td>
    <td>45.8</td>
    <td>42.7</td>
  </tr>
  <tr>
    <td>MASA-Detic</td>
    <td>50.6</td>
    <td>31.5</td>
    <td>50.6</td>
    <td>72.2</td>
    <td>60.1</td>
    <td>73.7</td>
    <td>46.5</td>
    <td>44.5</td>
  </tr>
  <tr>
    <td>MASA-G-DINO</td>
    <td>50.4</td>
    <td>31.2</td>
    <td>49.7</td>
    <td>72.8</td>
    <td>61.0</td>
    <td>74.3</td>
    <td>46.8</td>
    <td>45.0</td>
  </tr>
  <tr>
    <td>MASA-SAM-B</td>
    <td>49.4</td>
    <td>29.9</td>
    <td>48.0</td>
    <td>71.9</td>
    <td>59.5</td>
    <td>72.6</td>
    <td>46.2</td>
    <td>43.7</td>
  </tr>
  <tr>
    <td>HAT-MASA-R50</td>
    <td>54.3 (+3.5)</td>
    <td>36.1 (+4.5)</td>
    <td>54.1 (+5.3)</td>
    <td>73.7 (+2.1)</td>
    <td>62.4 (+3.5)</td>
    <td>75.0 (+3.3)</td>
    <td>46.4 (+0.6)</td>
    <td>44.4 (+1.7)</td>
  </tr>
  <tr>
    <td>HAT-MASA-Detic</td>
    <td>54.3 (+3.7)</td>
    <td>36.2 (+5.7)</td>
    <td>55.3 (+4.7)</td>
    <td>74.5 (+2.3)</td>
    <td>63.7 (+3.6)</td>
    <td>76.7 (+3.0)</td>
    <td>47.2 (+0.7)</td>
    <td>46.4 (+1.9)</td>
  </tr>
  <tr>
    <td>HAT-MASA-G-DINO</td>
    <td>53.9 (+3.5)</td>
    <td>35.7 (+4.5)</td>
    <td>54.5 (+4.8)</td>
    <td>74.7 (+1.9)</td>
    <td>64.1 (+3.1)</td>
    <td>77.1 (+2.8)</td>
    <td>47.5 (+0.7)</td>
    <td>46.7 (+1.7)</td>
  </tr>
  <tr>
    <td>HAT-MASA-SAM-B</td>
    <td>52.1 (+2.7)</td>
    <td>33.4 (+3.5)</td>
    <td>51.9 (+3.9)</td>
    <td>73.4 (+1.5)</td>
    <td>61.9 (+2.4)</td>
    <td>74.8 (+2.2)</td>
    <td>46.9 (+0.7)</td>
    <td>45.6 (+1.9)</td>
  </tr>
</table>

***NOTE:*** The results with the `HAT-` prefix incorporate our proposed History-Aware Transformation.

## Running Scripts

Our execution scripts follow the same format as [MASA](https://github.com/siyuanliii/masa). For more details, you can refer to [masa_benchmark_test.md](./masa_benchmark_test.md).

For example, if you want to run MASA-GroundingDINO on DanceTrack, you can use the following script:

```shell
OMP_NUM_THREADS=8 tools/dist_test.sh configs/masa-gdino/dance_test/masa_gdino_swinb_dance_test_yolox_dets_transfer.py ./saved_models/masa_models/gdino_masa.pth 8 --cfg-options model.tracker.use_transfer=False
```

If you want to run our HAT-MASA-GroundingDINO on DanceTrack, you can change the `use_transfer` parameter to `True`:

```shell
OMP_NUM_THREADS=8 tools/dist_test.sh configs/masa-gdino/dance_test/masa_gdino_swinb_dance_test_yolox_dets_transfer.py ./saved_models/masa_models/gdino_masa.pth 8 --cfg-options model.tracker.use_transfer=True
```

If you want to evaluate other models or other benchmarks, you should change the corresponding config file and the model path.

