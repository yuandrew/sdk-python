# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: temporal/api/enums/v1/reset.proto
"""Generated protocol buffer code."""

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import enum_type_wrapper

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n!temporal/api/enums/v1/reset.proto\x12\x15temporal.api.enums.v1*\xec\x01\n\x17ResetReapplyExcludeType\x12*\n&RESET_REAPPLY_EXCLUDE_TYPE_UNSPECIFIED\x10\x00\x12%\n!RESET_REAPPLY_EXCLUDE_TYPE_SIGNAL\x10\x01\x12%\n!RESET_REAPPLY_EXCLUDE_TYPE_UPDATE\x10\x02\x12$\n RESET_REAPPLY_EXCLUDE_TYPE_NEXUS\x10\x03\x12\x31\n)RESET_REAPPLY_EXCLUDE_TYPE_CANCEL_REQUEST\x10\x04\x1a\x02\x08\x01*\x97\x01\n\x10ResetReapplyType\x12"\n\x1eRESET_REAPPLY_TYPE_UNSPECIFIED\x10\x00\x12\x1d\n\x19RESET_REAPPLY_TYPE_SIGNAL\x10\x01\x12\x1b\n\x17RESET_REAPPLY_TYPE_NONE\x10\x02\x12#\n\x1fRESET_REAPPLY_TYPE_ALL_ELIGIBLE\x10\x03*n\n\tResetType\x12\x1a\n\x16RESET_TYPE_UNSPECIFIED\x10\x00\x12"\n\x1eRESET_TYPE_FIRST_WORKFLOW_TASK\x10\x01\x12!\n\x1dRESET_TYPE_LAST_WORKFLOW_TASK\x10\x02\x42\x82\x01\n\x18io.temporal.api.enums.v1B\nResetProtoP\x01Z!go.temporal.io/api/enums/v1;enums\xaa\x02\x17Temporalio.Api.Enums.V1\xea\x02\x1aTemporalio::Api::Enums::V1b\x06proto3'
)

_RESETREAPPLYEXCLUDETYPE = DESCRIPTOR.enum_types_by_name["ResetReapplyExcludeType"]
ResetReapplyExcludeType = enum_type_wrapper.EnumTypeWrapper(_RESETREAPPLYEXCLUDETYPE)
_RESETREAPPLYTYPE = DESCRIPTOR.enum_types_by_name["ResetReapplyType"]
ResetReapplyType = enum_type_wrapper.EnumTypeWrapper(_RESETREAPPLYTYPE)
_RESETTYPE = DESCRIPTOR.enum_types_by_name["ResetType"]
ResetType = enum_type_wrapper.EnumTypeWrapper(_RESETTYPE)
RESET_REAPPLY_EXCLUDE_TYPE_UNSPECIFIED = 0
RESET_REAPPLY_EXCLUDE_TYPE_SIGNAL = 1
RESET_REAPPLY_EXCLUDE_TYPE_UPDATE = 2
RESET_REAPPLY_EXCLUDE_TYPE_NEXUS = 3
RESET_REAPPLY_EXCLUDE_TYPE_CANCEL_REQUEST = 4
RESET_REAPPLY_TYPE_UNSPECIFIED = 0
RESET_REAPPLY_TYPE_SIGNAL = 1
RESET_REAPPLY_TYPE_NONE = 2
RESET_REAPPLY_TYPE_ALL_ELIGIBLE = 3
RESET_TYPE_UNSPECIFIED = 0
RESET_TYPE_FIRST_WORKFLOW_TASK = 1
RESET_TYPE_LAST_WORKFLOW_TASK = 2


if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    DESCRIPTOR._serialized_options = b"\n\030io.temporal.api.enums.v1B\nResetProtoP\001Z!go.temporal.io/api/enums/v1;enums\252\002\027Temporalio.Api.Enums.V1\352\002\032Temporalio::Api::Enums::V1"
    _RESETREAPPLYEXCLUDETYPE.values_by_name[
        "RESET_REAPPLY_EXCLUDE_TYPE_CANCEL_REQUEST"
    ]._options = None
    _RESETREAPPLYEXCLUDETYPE.values_by_name[
        "RESET_REAPPLY_EXCLUDE_TYPE_CANCEL_REQUEST"
    ]._serialized_options = b"\010\001"
    _RESETREAPPLYEXCLUDETYPE._serialized_start = 61
    _RESETREAPPLYEXCLUDETYPE._serialized_end = 297
    _RESETREAPPLYTYPE._serialized_start = 300
    _RESETREAPPLYTYPE._serialized_end = 451
    _RESETTYPE._serialized_start = 453
    _RESETTYPE._serialized_end = 563
# @@protoc_insertion_point(module_scope)
