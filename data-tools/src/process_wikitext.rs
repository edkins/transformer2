use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{anychar, satisfy},
    combinator::{recognize, map, value},
    multi::{many0, many1},
    sequence::{delimited, preceded},
    IResult,
};

fn plain(i: &str) -> IResult<&str, &str> {
    recognize(many1(satisfy(
        |c| matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | ' ' | '\n' | ',' | '.' | '(' | ')' | ':' | '-' | '!' | '/'),
    )))(i)
}

fn link_contents(i: &str) -> IResult<&str, &str> {
    recognize(many1(satisfy(|c| !matches!(c, ']' | '|' | '['))))(i)
}

fn url_contents(i: &str) -> IResult<&str, &str> {
    recognize(many1(satisfy(|c| !matches!(c, ']' | ' ' | '['))))(i)
}

fn image_contents(i: &str) -> IResult<&str, &str> {
    value("", many1(satisfy(|c| !matches!(c, ']' | '['))))(i)
}

fn image_items(i: &str) -> IResult<&str, &str> {
    value("", many1(alt((image_contents, plain_link, named_link))))(i)
}

fn template_contents(i: &str) -> IResult<&str, &str> {
    value("", many1(satisfy(|c| !matches!(c, '}' | '{' | '&'))))(i)
}

fn template_items(i: &str) -> IResult<&str, &str> {
    value("", many1(alt((template_contents, template, table, nowiki, reference, entity))))(i)
}

fn bold_or_italic(i: &str) -> IResult<&str, &str> {
    value("", alt((tag("'''"), tag("''"))))(i)
}

fn heading(i: &str) -> IResult<&str, &str> {
    value("", preceded(tag("="), many1(tag("="))))(i)
}

fn entity(i: &str) -> IResult<&str, &str> {
    alt((
        value("&", tag("&amp;")),
        value("<", tag("&lt;")),
        value(">", tag("&gt;")),
        value("\"", tag("&quot;")),
    ))(i)
}

fn nowiki_contents(i: &str) -> IResult<&str, &str> {
    recognize(many1(satisfy(|c| !matches!(c, '&'))))(i)
}

fn nowiki(i: &str) -> IResult<&str, &str> {
    delimited(tag("&lt;nowiki&gt;"), nowiki_contents, tag("&lt;/nowiki&gt;"))(i)
}

fn reference(i: &str) -> IResult<&str, &str> {
    delimited(tag("&lt;ref&gt;"), nowiki_contents, tag("&lt;ref&gt;"))(i)
}

fn image(i: &str) -> IResult<&str, &str> {
    value("", delimited(alt((tag("[[image:"), tag("[[Image:"))), image_items, tag("]]")))(i)
}

fn plain_link(i: &str) -> IResult<&str, &str> {
    map(delimited(tag("[["), link_contents, tag("]]")), |text| if text.contains(":") {""} else {text})(i)
}

fn named_link(i: &str) -> IResult<&str, &str> {
    let (i, _) = tag("[[")(i)?;
    let (i, _) = link_contents(i)?;
    let (i, _) = tag("|")(i)?;
    let (i, result) = link_contents(i)?;
    let (i, _) = tag("]]")(i)?;
    Ok((i, result))
}

fn url(i: &str) -> IResult<&str, &str> {
    recognize(preceded(alt((tag("http://"), tag("https://"))), url_contents))(i)
}

fn url_link(i: &str) -> IResult<&str, &str> {
    let (i, _) = tag("[")(i)?;
    let (i, _) = url(i)?;
    let (i, _) = tag(" ")(i)?;
    let (i, result) = link_contents(i)?;
    let (i, _) = tag("]")(i)?;
    Ok((i, result))
}

fn template(i: &str) -> IResult<&str, &str> {
    value("", delimited(tag("{{"), template_items, tag("}}")))(i)
}

fn table(i: &str) -> IResult<&str, &str> {
    value("", delimited(tag("{|"), template_items, tag("}")))(i)
}

fn character(i: &str) -> IResult<&str, &str> {
    recognize(anychar)(i)
}

fn item(i: &str) -> IResult<&str, &str> {
    alt((plain, bold_or_italic, heading, nowiki, reference, entity, image, plain_link, named_link, url_link, template, table, character))(i)
}

pub fn strip_wikitext(input: &str) -> Option<String> {
    if input.starts_with("#REDIRECT ") {
        None
    } else {
        let mut result = String::new();
        if let Ok((_, items)) = many0(item)(input) {
            for item in &items {
                result.push_str(item);
            }
        }
        Some(result)
    }
}
