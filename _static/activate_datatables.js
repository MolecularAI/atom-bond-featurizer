// Copyright (c) 2023 Varun Sharma
//
// SPDX-License-Identifier: MIT

$(document).ready( function () {
    $.extend( $.fn.dataTable.defaults,
        {
            "pageLength": 10,
            "language": {
                "lengthLabels": {
                    "-1": "Show all"
                }
            },
            "lengthMenu": [
                10,
                25,
                50,
                -1
            ],
            "dom": "Blfrtip",
            "buttons": [
                {
                    "extend": "colvis",
                    "text": "Show or hide columns"
                }
            ]
        },
    );

    $(`table.sphinx-datatable`).filter(':not(.dataTable)').DataTable(
        {},
    );
} );