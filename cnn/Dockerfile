# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


FROM tensorflow/tensorflow:1.13.2-py3
RUN pip install pandas
RUN pip install keras 
RUN pip install sklearn
ADD blerssi-cnn-adam.py  /opt/blerssi-cnn-adam.py
ADD iBeacon_RSSI_Labeled.csv /opt/iBeacon_RSSI_Labeled.csv
RUN chmod +x /opt/blerssi-cnn-adam.py  /opt/iBeacon_RSSI_Labeled.csv
CMD python /opt/blerssi-cnn-adam.py

