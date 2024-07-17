# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: temporal/api/cloud/identity/v1/message.proto
"""Generated protocol buffer code."""

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n,temporal/api/cloud/identity/v1/message.proto\x12\x1etemporal.api.cloud.identity.v1\x1a\x1fgoogle/protobuf/timestamp.proto"\x1d\n\rAccountAccess\x12\x0c\n\x04role\x18\x01 \x01(\t"%\n\x0fNamespaceAccess\x12\x12\n\npermission\x18\x01 \x01(\t"\x95\x02\n\x06\x41\x63\x63\x65ss\x12\x45\n\x0e\x61\x63\x63ount_access\x18\x01 \x01(\x0b\x32-.temporal.api.cloud.identity.v1.AccountAccess\x12Y\n\x12namespace_accesses\x18\x02 \x03(\x0b\x32=.temporal.api.cloud.identity.v1.Access.NamespaceAccessesEntry\x1ai\n\x16NamespaceAccessesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12>\n\x05value\x18\x02 \x01(\x0b\x32/.temporal.api.cloud.identity.v1.NamespaceAccess:\x02\x38\x01"Q\n\x08UserSpec\x12\r\n\x05\x65mail\x18\x01 \x01(\t\x12\x36\n\x06\x61\x63\x63\x65ss\x18\x02 \x01(\x0b\x32&.temporal.api.cloud.identity.v1.Access"p\n\nInvitation\x12\x30\n\x0c\x63reated_time\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x30\n\x0c\x65xpired_time\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp"\xb9\x02\n\x04User\x12\n\n\x02id\x18\x01 \x01(\t\x12\x18\n\x10resource_version\x18\x02 \x01(\t\x12\x36\n\x04spec\x18\x03 \x01(\x0b\x32(.temporal.api.cloud.identity.v1.UserSpec\x12\r\n\x05state\x18\x04 \x01(\t\x12\x1a\n\x12\x61sync_operation_id\x18\x05 \x01(\t\x12>\n\ninvitation\x18\x06 \x01(\x0b\x32*.temporal.api.cloud.identity.v1.Invitation\x12\x30\n\x0c\x63reated_time\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x36\n\x12last_modified_time\x18\x08 \x01(\x0b\x32\x1a.google.protobuf.Timestamp"U\n\rUserGroupSpec\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x36\n\x06\x61\x63\x63\x65ss\x18\x02 \x01(\x0b\x32&.temporal.api.cloud.identity.v1.Access"\x83\x02\n\tUserGroup\x12\n\n\x02id\x18\x01 \x01(\t\x12\x18\n\x10resource_version\x18\x02 \x01(\t\x12;\n\x04spec\x18\x03 \x01(\x0b\x32-.temporal.api.cloud.identity.v1.UserGroupSpec\x12\r\n\x05state\x18\x04 \x01(\t\x12\x1a\n\x12\x61sync_operation_id\x18\x05 \x01(\t\x12\x30\n\x0c\x63reated_time\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x36\n\x12last_modified_time\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.Timestamp"\x8d\x02\n\x0eServiceAccount\x12\n\n\x02id\x18\x01 \x01(\t\x12\x18\n\x10resource_version\x18\x02 \x01(\t\x12@\n\x04spec\x18\x03 \x01(\x0b\x32\x32.temporal.api.cloud.identity.v1.ServiceAccountSpec\x12\r\n\x05state\x18\x04 \x01(\t\x12\x1a\n\x12\x61sync_operation_id\x18\x05 \x01(\t\x12\x30\n\x0c\x63reated_time\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x36\n\x12last_modified_time\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.Timestamp"o\n\x12ServiceAccountSpec\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x36\n\x06\x61\x63\x63\x65ss\x18\x02 \x01(\x0b\x32&.temporal.api.cloud.identity.v1.Access\x12\x13\n\x0b\x64\x65scription\x18\x03 \x01(\t"\xfd\x01\n\x06\x41piKey\x12\n\n\x02id\x18\x01 \x01(\t\x12\x18\n\x10resource_version\x18\x02 \x01(\t\x12\x38\n\x04spec\x18\x03 \x01(\x0b\x32*.temporal.api.cloud.identity.v1.ApiKeySpec\x12\r\n\x05state\x18\x04 \x01(\t\x12\x1a\n\x12\x61sync_operation_id\x18\x05 \x01(\t\x12\x30\n\x0c\x63reated_time\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x36\n\x12last_modified_time\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.Timestamp"\xa0\x01\n\nApiKeySpec\x12\x10\n\x08owner_id\x18\x01 \x01(\t\x12\x12\n\nowner_type\x18\x02 \x01(\t\x12\x14\n\x0c\x64isplay_name\x18\x03 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x04 \x01(\t\x12/\n\x0b\x65xpiry_time\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x10\n\x08\x64isabled\x18\x06 \x01(\x08\x42\xac\x01\n!io.temporal.api.cloud.identity.v1B\x0cMessageProtoP\x01Z-go.temporal.io/api/cloud/identity/v1;identity\xaa\x02 Temporalio.Api.Cloud.Identity.V1\xea\x02$Temporalio::Api::Cloud::Identity::V1b\x06proto3'
)


_ACCOUNTACCESS = DESCRIPTOR.message_types_by_name["AccountAccess"]
_NAMESPACEACCESS = DESCRIPTOR.message_types_by_name["NamespaceAccess"]
_ACCESS = DESCRIPTOR.message_types_by_name["Access"]
_ACCESS_NAMESPACEACCESSESENTRY = _ACCESS.nested_types_by_name["NamespaceAccessesEntry"]
_USERSPEC = DESCRIPTOR.message_types_by_name["UserSpec"]
_INVITATION = DESCRIPTOR.message_types_by_name["Invitation"]
_USER = DESCRIPTOR.message_types_by_name["User"]
_USERGROUPSPEC = DESCRIPTOR.message_types_by_name["UserGroupSpec"]
_USERGROUP = DESCRIPTOR.message_types_by_name["UserGroup"]
_SERVICEACCOUNT = DESCRIPTOR.message_types_by_name["ServiceAccount"]
_SERVICEACCOUNTSPEC = DESCRIPTOR.message_types_by_name["ServiceAccountSpec"]
_APIKEY = DESCRIPTOR.message_types_by_name["ApiKey"]
_APIKEYSPEC = DESCRIPTOR.message_types_by_name["ApiKeySpec"]
AccountAccess = _reflection.GeneratedProtocolMessageType(
    "AccountAccess",
    (_message.Message,),
    {
        "DESCRIPTOR": _ACCOUNTACCESS,
        "__module__": "temporal.api.cloud.identity.v1.message_pb2",
        # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.AccountAccess)
    },
)
_sym_db.RegisterMessage(AccountAccess)

NamespaceAccess = _reflection.GeneratedProtocolMessageType(
    "NamespaceAccess",
    (_message.Message,),
    {
        "DESCRIPTOR": _NAMESPACEACCESS,
        "__module__": "temporal.api.cloud.identity.v1.message_pb2",
        # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.NamespaceAccess)
    },
)
_sym_db.RegisterMessage(NamespaceAccess)

Access = _reflection.GeneratedProtocolMessageType(
    "Access",
    (_message.Message,),
    {
        "NamespaceAccessesEntry": _reflection.GeneratedProtocolMessageType(
            "NamespaceAccessesEntry",
            (_message.Message,),
            {
                "DESCRIPTOR": _ACCESS_NAMESPACEACCESSESENTRY,
                "__module__": "temporal.api.cloud.identity.v1.message_pb2",
                # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.Access.NamespaceAccessesEntry)
            },
        ),
        "DESCRIPTOR": _ACCESS,
        "__module__": "temporal.api.cloud.identity.v1.message_pb2",
        # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.Access)
    },
)
_sym_db.RegisterMessage(Access)
_sym_db.RegisterMessage(Access.NamespaceAccessesEntry)

UserSpec = _reflection.GeneratedProtocolMessageType(
    "UserSpec",
    (_message.Message,),
    {
        "DESCRIPTOR": _USERSPEC,
        "__module__": "temporal.api.cloud.identity.v1.message_pb2",
        # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.UserSpec)
    },
)
_sym_db.RegisterMessage(UserSpec)

Invitation = _reflection.GeneratedProtocolMessageType(
    "Invitation",
    (_message.Message,),
    {
        "DESCRIPTOR": _INVITATION,
        "__module__": "temporal.api.cloud.identity.v1.message_pb2",
        # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.Invitation)
    },
)
_sym_db.RegisterMessage(Invitation)

User = _reflection.GeneratedProtocolMessageType(
    "User",
    (_message.Message,),
    {
        "DESCRIPTOR": _USER,
        "__module__": "temporal.api.cloud.identity.v1.message_pb2",
        # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.User)
    },
)
_sym_db.RegisterMessage(User)

UserGroupSpec = _reflection.GeneratedProtocolMessageType(
    "UserGroupSpec",
    (_message.Message,),
    {
        "DESCRIPTOR": _USERGROUPSPEC,
        "__module__": "temporal.api.cloud.identity.v1.message_pb2",
        # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.UserGroupSpec)
    },
)
_sym_db.RegisterMessage(UserGroupSpec)

UserGroup = _reflection.GeneratedProtocolMessageType(
    "UserGroup",
    (_message.Message,),
    {
        "DESCRIPTOR": _USERGROUP,
        "__module__": "temporal.api.cloud.identity.v1.message_pb2",
        # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.UserGroup)
    },
)
_sym_db.RegisterMessage(UserGroup)

ServiceAccount = _reflection.GeneratedProtocolMessageType(
    "ServiceAccount",
    (_message.Message,),
    {
        "DESCRIPTOR": _SERVICEACCOUNT,
        "__module__": "temporal.api.cloud.identity.v1.message_pb2",
        # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.ServiceAccount)
    },
)
_sym_db.RegisterMessage(ServiceAccount)

ServiceAccountSpec = _reflection.GeneratedProtocolMessageType(
    "ServiceAccountSpec",
    (_message.Message,),
    {
        "DESCRIPTOR": _SERVICEACCOUNTSPEC,
        "__module__": "temporal.api.cloud.identity.v1.message_pb2",
        # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.ServiceAccountSpec)
    },
)
_sym_db.RegisterMessage(ServiceAccountSpec)

ApiKey = _reflection.GeneratedProtocolMessageType(
    "ApiKey",
    (_message.Message,),
    {
        "DESCRIPTOR": _APIKEY,
        "__module__": "temporal.api.cloud.identity.v1.message_pb2",
        # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.ApiKey)
    },
)
_sym_db.RegisterMessage(ApiKey)

ApiKeySpec = _reflection.GeneratedProtocolMessageType(
    "ApiKeySpec",
    (_message.Message,),
    {
        "DESCRIPTOR": _APIKEYSPEC,
        "__module__": "temporal.api.cloud.identity.v1.message_pb2",
        # @@protoc_insertion_point(class_scope:temporal.api.cloud.identity.v1.ApiKeySpec)
    },
)
_sym_db.RegisterMessage(ApiKeySpec)

if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    DESCRIPTOR._serialized_options = b"\n!io.temporal.api.cloud.identity.v1B\014MessageProtoP\001Z-go.temporal.io/api/cloud/identity/v1;identity\252\002 Temporalio.Api.Cloud.Identity.V1\352\002$Temporalio::Api::Cloud::Identity::V1"
    _ACCESS_NAMESPACEACCESSESENTRY._options = None
    _ACCESS_NAMESPACEACCESSESENTRY._serialized_options = b"8\001"
    _ACCOUNTACCESS._serialized_start = 113
    _ACCOUNTACCESS._serialized_end = 142
    _NAMESPACEACCESS._serialized_start = 144
    _NAMESPACEACCESS._serialized_end = 181
    _ACCESS._serialized_start = 184
    _ACCESS._serialized_end = 461
    _ACCESS_NAMESPACEACCESSESENTRY._serialized_start = 356
    _ACCESS_NAMESPACEACCESSESENTRY._serialized_end = 461
    _USERSPEC._serialized_start = 463
    _USERSPEC._serialized_end = 544
    _INVITATION._serialized_start = 546
    _INVITATION._serialized_end = 658
    _USER._serialized_start = 661
    _USER._serialized_end = 974
    _USERGROUPSPEC._serialized_start = 976
    _USERGROUPSPEC._serialized_end = 1061
    _USERGROUP._serialized_start = 1064
    _USERGROUP._serialized_end = 1323
    _SERVICEACCOUNT._serialized_start = 1326
    _SERVICEACCOUNT._serialized_end = 1595
    _SERVICEACCOUNTSPEC._serialized_start = 1597
    _SERVICEACCOUNTSPEC._serialized_end = 1708
    _APIKEY._serialized_start = 1711
    _APIKEY._serialized_end = 1964
    _APIKEYSPEC._serialized_start = 1967
    _APIKEYSPEC._serialized_end = 2127
# @@protoc_insertion_point(module_scope)